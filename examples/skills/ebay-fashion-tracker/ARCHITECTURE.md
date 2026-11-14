# eBay Fashion Tracker: architecture

This is the design of the skill. The code in this folder is version 0.1 of it; [section 17](#17-what-version-01-implements) says which parts exist and which are still design.

The skill is `ebay-fashion-tracker`; its command-line tool is `eft`.

## 1. What it does

You give Claude Code a photo, an eBay link, a retailer link or a description of a fashion item. Open models on Crusoe turn that into a typed item spec, and you confirm the size and price ceiling. A background poller then searches eBay through its official API, has the Crusoe models judge each new listing against the spec, and sends you a Slack card or phone push when one matches, with price and bid state. You bid or buy by tapping through to eBay. The tool never does.

## 2. Shape

```
 You ── photo / link / text ──▶ Claude Code + skill (SKILL.md)
                                      │ runs
                                      ▼
                                   eft CLI ◀── same code ──▶ eft tick  (poller)
                                      │                        launchd on the Mac to start,
                                      │                        systemd on a host later
                                      ▼                           │
                 state: SQLite + your reference photos            ├──▶ eBay Browse API (search, listing detail)
                 secrets: .env or the environment                 ├──▶ Crusoe inference (open models)
                                                                  ├──▶ Slack DM card (threads, edits)
                                                                  ├──▶ Phone push (6-hour expiry)
                                                                  └──▶ Heartbeat service

 eBay ── account-deletion notices ──▶ Cloudflare Worker ◀── signed phone-action links (mute, not a match, star)
```

| Piece | Runs where | Job |
|---|---|---|
| Skill (`SKILL.md`) | Claude Code session | Conversation and clarifying questions; calls `eft`; may fetch retailer pages, never eBay pages |
| `eft` CLI | Your machine | All logic. One codebase used by Claude and by the poller. Python 3.9+, standard library only |
| Poller (`eft tick`) | launchd on a Mac first; the same command under cron or systemd on an always-on host later | Search, match, alert, refresh, expire, purge |
| State | `~/Library/Application Support/ebay-fashion-tracker/` on macOS, `~/.local/share/ebay-fashion-tracker/` elsewhere | One SQLite file and your reference photos |
| Secrets | `.env` in the skill folder, or the environment | eBay keyset, Crusoe key, Slack token, push keys. Typed by you, never through Claude, and never committed |
| Worker | Cloudflare, free tier | Answers eBay's account-deletion challenge; holds a small inbox of phone actions |
| Heartbeat | External dead-man service | Tells you on your phone when polling has stopped |

## 3. Main decisions

| Question | Choice | Why |
|---|---|---|
| Where polling runs | A plain script, not a Claude scheduler | Claude's `/loop` and cron tasks need an open session, cost a Claude turn per tick and expire after seven days. A script costs no Claude tokens |
| Laptop or server | Laptop first, written so the same `eft tick` moves to a host by copying the state folder | Shortest path to a first alert. Section 11 lists what the laptop cannot do |
| How eBay is read | Browse API only, with an application token | eBay's User Agreement bans automated access to ebay.com by "LLM-driven bots". The skill forbids Claude from fetching any ebay.com page, short links included |
| Bids | Reported, never placed | The bidding API is closed to personal tools, and the agreement bans order flows without human review |
| Who decides a match | Code, from per-field model verdicts | The models supply evidence; a fixed rule makes the call. No model-reported confidence numbers |
| Which Crusoe models | Chosen at run time from the live catalog, per role, with fallbacks | Ids are case-sensitive, the docs misprint some, and retired ids are silently redirected for 90 days |
| eBay key activation | A Worker answers deletion notices, and no seller identifier is ever stored | Production keys stay disabled without it. The alternative, declaring "we store no eBay data", would be untrue for a tracker |
| Alert channels | Slack DM card plus phone push | Slack gives threads and edits. Push works on day one and carries every safety alert, so nothing depends on Slack approval |

## 4. Item spec

A watch is a typed spec, not a keyword string.

- `brand`, `model`, `style_codes[]`
- `colorway` {official name, nicknames, colors}
- `size` {value, system: US, UK, EU, JP, CM, alpha or waist/inseam; width}
- `department`: men, women, unisex, or kids (GS, PS, TD)
- `conditions[]`, `price_ceiling` {amount, currency, shipping included or not}, `buying_options[]`
- `must_have[]`, `must_not[]`, `visual_signature[]` (what the item looks like, in words)
- `reference_images[]`: yours or a retailer's, never eBay's
- For every field: **required**, **optional** or **not applicable**, and where its value came from

Rules that go with it:

- **Required fields are per watch.** A vintage or unbranded piece marks brand and style code "not applicable" and can still alert.
- **Style codes and nicknames never come from a model's memory.** They are accepted only from text read off your photo, your own input, a retailer page, or a confirming search.
- **Size is proven only by what the listing states.** The listing's own size in your size system, or an equivalent the seller gave, counts as a match. A size the tool had to convert counts as "unknown". Italian, French, numeric dress and measured vintage sizing are always "unknown" in version 1.

## 5. Matching funnel

Each new listing passes through stages, cheapest first. Crusoe calls are plentiful and eBay calls are scarce, so the models run before an eBay detail call is spent.

| Stage | What happens | Cost |
|---|---|---|
| 0. Rules | Drop already-seen ids, muted sellers, a conflicting style code, a clear department conflict. A listing over your ceiling is parked, not rejected | Free |
| 1. Title triage | Text model, 10 titles per call: yes, maybe or no, each with a reason code. Only a reasoned "no" is dropped. Every id must come back exactly once | Crusoe |
| 2. Photo pre-check | For a "maybe": compare the search-result photo with your reference, or with the visual signature when there is no reference | Crusoe |
| 3. Detail fetch | Fetch the full listing. In a multi-size listing, find the in-stock variation in your size. A size in the title never rejects a multi-size listing | 1 to 2 eBay calls |
| 4. Spec check | Deterministic check of size, style code and stock | Free |
| 5. Detail verify | Text model reads item specifics and description and returns match, mismatch or unknown per field, each with a verbatim quote, plus flags (replica, custom, lot, box only). Quotes are checked to be real substrings of the listing | Crusoe |
| 6. Tag read | Only for fields still unknown: read the size tag or box label in the listing's other photos, one image per call, four at most | Crusoe |

**Decision rule**

- **Alert** when every required field is a match and none is a mismatch.
- **Reject** on any mismatch. If the mismatch can change (over your ceiling, your size out of stock), park the listing and recheck it on the periodic sweep.
- **Maybe** for anything else. It goes to a daily digest that names the unproven field. An auction that ends before the next digest is sent at once, marked "unconfirmed".

**Details that matter**

- **Failed model calls.** A listing with no valid verdict is retried up to three times, then moved to "maybe". It never alerts, is never silently dropped, and never blocks the rest of the watch.
- **Trap words.** "Replica", "UA", "lot", "custom" and "laces" are judged in context by the model, not banned as tokens, and are ignored when they are part of the watch's own brand or model (Margiela Replica, Under Armour, Levi's Lot 501).
- **Authenticity.** Photo agreement is evidence of model and colorway only. Every card says "not authenticated".
- **Feedback.** "A17 isn't a match, it's the kids version" makes a text model propose a spec change. The change is replayed over current candidates ("removes 3 alerts, keeps 2") and applied only when you approve.
- **Misses.** "You missed this one" plus a link shows which stage dropped the listing, from a stored trace or a live replay.

## 6. Crusoe model roles

Ids are resolved from the live catalog by capability and a case-insensitive name pattern. The names below are today's first choices, not hardcoded values.

| Role | First choice | Fallback | Used when |
|---|---|---|---|
| Read your photo into a draft spec, with any visible text | Gemma 4 31B | Qwen3.8 27B | Adding from an image |
| Parse text into a spec; plan eBay queries from real category and aspect names | gpt-oss-120b | DeepSeek V4.1 Flash | Adding, feedback |
| Title triage | gpt-oss-120b, low effort | DeepSeek V4.1 Flash | Every tick, new listings only |
| Detail verify | gpt-oss-120b | DeepSeek V4.1 Flash | Shortlisted listings |
| Photo compare and tag read | Gemma 4 31B | Qwen3.8 27B as second opinion | Maybes and unproven fields |

Nemotron 3.5 Lightning is cheaper for triage but measured at about 16 seconds with reasoning on. It takes the role only if the first-run probe shows under 3 seconds with reasoning off and it passes the gate below.

**Call rules**

- Send only `model`, `messages`, `max_tokens`, `temperature`, `response_format` and `reasoning_effort`, using values the probe proved for that model. Unknown parameters can return 403.
- `max_tokens` is 1,024 (2,048 for gpt-oss, whose reasoning cannot be turned off). An empty answer with `finish_reason: "length"` is retried once at double.
- JSON is validated in code, repaired once, then sent to the fallback model. Server-side schema enforcement is not assumed.
- Photo calls are designed for two images at most until the probe shows more are accepted.
- Compare the model that served each response with the one requested. On a mismatch, re-run the gate before trusting the replacement.
- 401 and 403 are never retried and raise an immediate phone alert.
- Listing images come only from URLs the API returned, on eBay's image host. They are held in memory, converted to a 1,024-pixel JPEG, sent as base64 and never stored.
- Seller text is stripped of HTML and passed as delimited data. Model output is enums and quotes only. Links on cards come from the API, never from a model.
- If Crusoe is down, only exact style-code matches alert, labelled "not model-checked".

**Quality gate.** About 40 hand-written listing records covering every trap, plus photo pairs from your own and retailer photos. A model holds a role only while the pipeline passes it. The gate re-runs on any model or prompt change. It contains no live eBay listings, because the licence bars using eBay content for machine learning. In production the signal is your feedback: a per-watch count of "not a match" and per-watch funnel counts.

**Cost.** Estimated at roughly $2 to $4 a month in model calls at 25 watches. That figure is a planning estimate, not a measurement; the call ledger gives the real one.

## 7. Poller

One tick:

1. Take a lock, read secrets, drain the phone-action inbox and the unsent-alert outbox.
2. Expire first: strip price fields from any message whose data is over six hours old, and purge ended listings.
3. Refresh tracked listings that are due, nearest end first, and thread any change.
4. Search each due watch: one query, the buying-options filter always set (without it eBay returns no auctions), and a start-date window running from the last successful tick minus 15 minutes. A window with more than 200 results is split by time and continued across ticks.
5. Run the funnel. Alert rows are written in the same transaction that advances the window, so a crash produces a late alert, never a lost or duplicate one.
6. About every six hours per watch, run an unwindowed sweep. One call refreshes price and bids for known listings, catches relists, and rechecks parked listings against your ceiling.
7. Send the outbox. Ping the heartbeat only if every eBay, Crusoe and send call in this tick succeeded.

**eBay call budget.** eBay allows 5,000 calls a day. The tool stops itself at 4,800, keeps 400 in reserve for adds and retries, and caps each watch at 25 non-search calls a day (8 detail fetches, 12 tracked refreshes, 4 sweeps, 1 parked recheck). When a cap is hit, work waits, soonest-ending first.

| Watches | Search cadence | Searches | Per-watch caps | Reserve | Total |
|---|---|---|---|---|---|
| 10 | every 5 min | 2,880 | 250 | 400 | 3,530 (71%) |
| 25 | every 10 min | 3,600 | 625 | 400 | 4,625 (93%) |
| 50 | 10 hot at 10 min, 40 at 40 min | 2,880 | 1,250 | 400 | 4,530 (91%) |

- Watches for one model in different sizes share a search.
- A single pinned listing costs 1,440 divided by its check interval in minutes.
- Time to detect a new listing is the cadence plus eBay's indexing delay, reported as seconds to about five minutes.
- More than about 50 watches needs eBay's Application Growth Check.
- These are planning figures. A ledger by purpose replaces them with measurements.

## 8. Alerts

**Card contents**

- Watch name and a short code such as A17.
- A proof line: "Size US 10 (item specifics) · DZ5485-612 (title) · colorway confirmed by photo".
- Price and shipping. For auctions: current bid, bid count, minimum next bid, reserve met, and "scheduled end 21:14, may extend".
- Condition, "as of 14:02", "not authenticated", and the listing link from the API.
- "Tap Watch on eBay" so eBay's own app sends its ending-soon reminders.
- Never on a card: a deal score, a price opinion, or anything from another retailer.

**Urgency**

| Tier | Delivery | What qualifies |
|---|---|---|
| P1 | May break quiet hours; starred or hot watches only; daily cap | Proven Buy It Now match within your ceiling; starred auction near its end; pinned listing back in stock |
| P2 | Instant in waking hours | Other proven matches; a parked listing reaching your ceiling |
| P3 | Morning digest | Maybes, bid movement, system notes |

**Mechanics**

- **No duplicates.** A unique key per event is written before sending, and an outbox retries failures.
- **Bursts.** More than three matches for one watch collapse into one message. After a long sleep, the backlog collapses and already-sold listings are dropped.
- **Quiet hours.** 22:00 to 08:00 by default. Held alerts are re-fetched before release, or dropped if the listing is gone.
- **Expiry.** The location of every message is stored. At six hours, price fields are stripped from cards, replies and digests alike. Phone pushes carry a six-hour expiry. When a listing ends, every message becomes "Ended" and its data is purged.
- **Auction reminders.** Slack messages with no price data are scheduled for 60 and 10 minutes before the scheduled end, so they fire with the Mac asleep. When the poller is awake it swaps them for a reply with live bids.
- **Safety alerts.** These go to phone push at once: Crusoe key rejected, eBay keyset disabled, Slack token revoked, quota at 80 percent, a model swapped underneath a role.
- **Phone actions.** Signed links for Mute, Not a match and Star open a confirm page on the Worker. The poller collects them each tick.

## 9. Compliance rules built in

- No fetch, curl or browser access to ebay.com by Claude or by `eft`. Links are parsed locally. For short links the user is asked for the full URL.
- No bidding, offers or purchases. No bid amounts or price opinions from a model.
- Listing data shown to you is at most six hours old and always carries its "as of" time.
- Data for ended listings is deleted. Rejected listings are kept as an id and a verdict code only, with no title or price.
- No seller usernames stored. Relists are labelled from the live listing's own dates, not from retained history.
- eBay content is used for inference only: no fine-tuning, no evaluation sets, no price statistics, no deal scoring.
- eBay data is never shown beside another retailer's data.
- `eft` prints as little listing data as possible into the Claude Code transcript, which is stored on disk.
- `eft uninstall` unloads the poller and deletes the database and state folder. The licence requires destroying eBay content within ten days of stopping. Removing the Worker and revoking keys are manual steps.

## 10. Data model

| Table | Holds |
|---|---|
| `watch` | Spec, query plan (category, real aspect names, query), cadence tier, search window, status |
| `candidate` | Listing id, state (pending, parked, maybe, alerted, rejected, unavailable), reason code, purge date. Title, price and image only while tracked |
| `verdict` | Funnel trace per candidate: stage, model that served, per-field result, rule fired. Evidence quotes are purged when the listing ends |
| `alert` | Dedup key, tier, status, "as of", and the location of every message sent |
| `feedback` | Your words, the reason code, the approved spec change |
| `ledger` | eBay and Crusoe calls by day and purpose |
| `capabilities` | Probe results and the current model for each role |

## 11. Laptop mode and host mode

Laptop mode is the starting point. These are its known limits:

| Limit on a laptop | Effect | Fixed by a host |
|---|---|---|
| The poller stops when the lid is closed | New-listing alerts arrive late. Nothing is lost: the search window resumes on wake | Yes |
| No live bids at 10 minutes to go while asleep | You get the data-free Slack reminder and eBay's own Watch push instead | Yes |
| A Slack card cannot be stripped while asleep | A card can show a price older than six hours until wake. It always shows its "as of" time | Yes |
| The heartbeat cannot tell asleep from broken | The alarm threshold has to be about a day | Yes, 15 minutes |

Host mode is the same `eft tick` under cron or a systemd timer on a small always-on machine, with the state folder copied over. It costs about what a small virtual machine costs, and it adds a server to patch and a second place holding every secret.

## 12. Skill layout

```
ebay-fashion-tracker/
  SKILL.md
  scripts/eft
  scripts/eft_core/   config, http, store, spec, rules, links, errors,
                      crusoe, ebay, prompts, plan, funnel, alerts, tick, doctor, cli
  references/         spec-schema, ebay-rules, troubleshooting
  deploy/             launchd plist template, Slack app manifest, Worker source
  tests/              unit tests against fake API responses
```

`SKILL.md` tells Claude to: route everything through `eft`; never open ebay.com; never bid or give price advice; ask for a file path when an image was pasted (pasted images have no path a script can read); always confirm size, size system and ceiling; show the dry run before activating a watch; run `eft inbox` at the start of a session.

What you can say: "track this" with a photo, link or description; "what am I tracking"; "A17 isn't a match because…"; "you missed this one" with a link; "why did A17 match"; "star A17"; "make this one hot"; "pause" or "remove"; "status"; "run doctor".

## 13. Setup order

| Step | Wait | Works afterwards |
|---|---|---|
| 1. Start eBay developer registration and the Slack app request | A day to more than a week each | |
| 2. Crusoe API key in `.env`; `eft doctor crusoe` | Minutes | Photos and text become draft watches |
| 3. Phone push and heartbeat; test alert sent by the poller itself, not the terminal | 10 minutes | Delivery path proven |
| 4. Deploy the Worker | 20 minutes | Deletion endpoint live |
| 5. eBay approves: create the production keyset, subscribe it to the Worker, `eft doctor ebay` | Minutes | Drafts activate; alerts by phone push |
| 6. Slack approved | Minutes | Cards, threads, reminders |
| 7. `eft doctor sleep` | 15 minutes | Wake and catch-up behaviour checked |

`eft status` shows which step is waiting. eBay registrations have been reported stuck or auto-rejected, so step 1 comes first.

## 14. First-run probes

Nothing was tested live during design. `eft doctor` settles these before any watch goes active:

- **eBay:** does a new keyset reach Browse at all; do auctions come back from a keyword-only search; does the start-date window pick up relists; what "price" means on an auction search result; does a multi-size listing give size and stock per variation; real indexing delay.
- **Crusoe:** exact live model ids; images accepted per request; whether JSON schemas are enforced; which reasoning settings each model accepts; triage latency with reasoning off.
- **Slack:** do eBay image URLs render in a card.
- **Your machine:** `.env` and network access from Claude's sandboxed shell; the poller's own delivery path; wake behaviour.
- **eBay image search:** five labelled photos. It is used as a naming hint only if it passes.

Claude Code's sandbox often denies reads of `.env` files. When it does, the variables have to be exported in the shell, or `eft` run from your own terminal. A planned alternative is for the in-session CLI to hand requests to the poller through a queue folder, so that it needs neither secrets nor network.

## 15. Decisions for you

1. **Laptop or host.** How often is your Mac closed when you would want an alert? Laptop mode is the default; section 11 is the trade.
2. **Which Slack.** A personal free workspace installs in minutes. A work workspace may need admin approval, and puts a shopping bot in work Slack.
3. **Phone channel.** Pushover (a one-time purchase, expiring messages, priorities) is the recommendation. Telegram is free. Real SMS needs carrier registration even to text yourself, which takes days and has monthly fees.
4. **What goes to Crusoe.** Listing text, listing photos and your own photos are sent for inference. Check the data-handling terms that apply to your account, and decide between a work and a personal key.
5. **Strictness.** Should "size not stated" go to the digest (recommended) or be dropped?
6. **Marketplace.** US only, or others as well. Each extra marketplace adds a full set of searches against the same daily budget.

## 16. Cut from version 1

- Bidding, Best Offer and purchase calls: closed to personal tools and against the agreement for an agent.
- Real SMS: carrier registration to text yourself.
- iMessage from the Mac: reported unreliable from a background job.
- Slack buttons: need an always-running listener and a second approval. Signed links replace them.
- Embeddings: Crusoe serves none.
- eBay image search inside the poll loop: unproven for fashion.
- Sold-price comparisons and deal scores: barred by eBay's licence.
- eBay's watchlist API: newly released, and whether an ordinary keyset can use it is unknown. A manual tap on Watch replaces it.

## 17. What version 0.1 implements

The sections above describe the design. This is what the code in this folder does today.

| Area | In version 0.1 | Not yet |
|---|---|---|
| Item spec | All fields, per-field required / optional / not applicable, validation, feedback patches | `conditions` and `must_have` are recorded but not enforced |
| Adding a watch | From a photo, a description, or an eBay item, product or search link (parsed, never fetched) | Tracking one exact listing (a pinned watch) |
| Matching funnel | Stages 0 to 6 and the decision rule; quotes checked against listing text; item groups resolved to the wanted size | Photo check against the visual signature when there is no reference photo |
| Crusoe models | Role-based selection from the live catalog with fallbacks, served-model check, JSON validation with one repair, client-side rate limit | Reasoning-effort control per model; the quality-gate evaluation set |
| eBay access | Application token, search, item, legacy id, item group; a daily cap, a per-watch daily allowance for refresh and detail calls, and a ledger by purpose | Image search |
| Poller | Windowed search, six-hourly sweep, tracked-listing refresh, ended and unavailable handling, parked rechecks, style-code-only mode when the models are down, safety notices, heartbeat | Relist threading; a packaged always-on host mode |
| Alerts | Slack direct-message cards with threads, six-hour stripping and "Ended" rewrites; Pushover with a six-hour expiry; quiet hours; a daily digest | Slack scheduled reminders; phone-action links; a daily cap on urgent alerts |
| Setup and upkeep | `eft doctor` for env, Crusoe, eBay and alerts; launchd install and a cron line elsewhere; `eft uninstall`; the deletion-notice Worker | `eft doctor sleep`; the queue-folder handoff for sandboxed sessions |

Everything in the middle column is covered by unit tests that run offline against fake API responses. None of it has yet been run against the live APIs; see the Status section of the README.
