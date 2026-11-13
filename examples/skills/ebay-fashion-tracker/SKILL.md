---
name: ebay-fashion-tracker
description: Track fashion items on eBay and get pinged when a matching listing appears. Use this whenever the user wants to find, watch or get alerts for clothing, sneakers, bags or accessories on eBay, starting from a photo, an eBay link, a retailer link or a description. Also use it for "what am I tracking", "any new listings", "what's the price or bid on that one", "why did this match", "that's not a match", or anything about pausing, starring or removing a tracked item. Matching runs on open models served by Crusoe Managed Inference; eBay is read only through its official API.
---

# eBay Fashion Tracker

You are the front end. The engine is the `eft` command, which builds an item spec with open models on Crusoe, searches eBay through the Browse API, judges each listing against the spec, and sends alerts. A background poller runs the same command on a schedule.

Run it as:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/eft" --json <command>
```

`--json` gives you one JSON document to read. Leave it off when the user wants to see the plain output.

## Rules

These come from eBay's User Agreement and API License Agreement. They are not preferences.

1. **Never open an eBay page.** No WebFetch, curl or browser on any `ebay.*` URL, `ebay.us` or `ebay.to` link, not even to "just check". Pass the link to `eft add --url`; it reads the link as text and looks the listing up through the API. For a short link, ask the user for the full URL.
2. **Never bid, make an offer or buy**, and never suggest a bid amount or say whether a price is good. Report the price and bid state; the user acts on eBay.
3. **Match decisions come from `eft`.** Relay them and explain them from the proof line and the trace. Do not overrule them from your own reading of a title.
4. **Show listing data only from `eft` output, with its "as of" time.** Never place it beside another retailer's price.
5. **Never read, print or ask the user to paste a key.** Keys live in `.env` or the environment. If one is missing, name the variable to set.
6. **Never call a listing authentic.** The tool cannot establish that, and its cards say "not authenticated".

## First use in a session

Run `eft --json status`. It reports which keys are configured (by name only), the watches, the last poll, eBay calls used today and any setup step still waiting. If something is missing, run `eft doctor env` (or `crusoe`, `ebay`, `alerts`) and tell the user what to do next. Setup detail is in `references/troubleshooting.md`.

If `status` says the `.env` file is unreadable, Claude Code's sandbox is blocking it. Tell the user to either export the variables in their shell profile or run `eft` from their own terminal.

## Adding something to track

Pick the input the user gave you:

| The user gives | Do this |
|---|---|
| A photo | You need a file path. A pasted image has none, so ask them to drag the file in or type its path. Then `eft add --image PATH --note "what they said about it"` |
| An eBay link | `eft add --url "URL"`. It drafts a watch for that item in general; following one exact listing is not supported yet, so say so if that is what they wanted |
| Another retailer's link | Fetch that page yourself (never an eBay page), pull out brand, model, style code and color, then `eft add --text "..."`. Carry no retailer price across unless the user names it as their ceiling |
| A description | `eft add --text "..."` |

`add` creates a **draft** watch and returns the spec the model drafted, its open questions and any problems. Then:

1. Show the user the one-line summary of the spec.
2. Ask about anything unknown. Always confirm **size, size system (US, UK, EU, alpha), department and price ceiling**; a photo cannot tell you these. A style code is only trusted when it was read from the photo, typed by the user or taken from a retailer page.
3. Apply the answers: `eft watch set ID size="US 10" department=men price_ceiling=180`.
4. Preview: `eft dryrun ID`. Show a few of the decisions and ask whether they look right.
5. Activate: `eft watch activate ID`. It refuses while the spec still has problems and says which.

Field meanings and the per-field `required | optional | na` setting are in `references/spec-schema.md`. Use `na` for fields that do not apply, such as brand on an unbranded vintage piece.

## Day to day

| The user says | Command |
|---|---|
| "what am I tracking" | `eft watch list` |
| "anything new" | `eft inbox` |
| "A17 isn't a match, it's the kids version" | `eft feedback A17 --reason "..."`, show the proposed change and how many candidates it affects, then `eft feedback --apply N` once they agree |
| "why did A17 match" / "why was this dropped" | `eft explain A17` |
| "star A17" | `eft star A17` (starred auctions get ending reminders) |
| "make the Jordans a priority" | `eft watch hot ID` |
| "pause" / "remove" | `eft watch pause ID`, `eft watch remove ID` |
| "check now" | `eft tick` |

## What the decisions mean

| Decision | Meaning |
|---|---|
| `alert` | Every required field was proven from the listing. The proof line says where each was proven |
| `maybe` | Nothing contradicted the spec, but a required field is unproven. Named in the digest |
| `parked` | Matches, but over the ceiling or the size is out of stock. Rechecked automatically |
| `reject` | A field contradicts the spec, or the listing is a replica, lot, custom or parts |
| `pending` | A model call failed. Retried, then moved to `maybe` |

## The background poller

`eft agent install` sets up a scheduled job that runs `eft tick`. It changes the user's machine, so ask before running it. `eft agent status` shows whether it is loaded; `eft agent uninstall` removes it. On a laptop the poller stops while the lid is closed; nothing is lost, alerts arrive when it wakes.

`eft uninstall` removes the job and deletes all stored data. Ask first.

## References

- `references/spec-schema.md`: every spec field and how sizes are proven.
- `references/ebay-rules.md`: what the eBay agreements require and how the tool complies.
- `references/troubleshooting.md`: setup order, key problems, sandbox notes, common errors.
