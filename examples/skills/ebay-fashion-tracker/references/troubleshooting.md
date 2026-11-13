# Setup order and troubleshooting

## Setup order

Some steps wait on other people, so start them first.

| Step | Wait | What works afterwards |
|---|---|---|
| 1. Register at [developer.ebay.com](https://developer.ebay.com/) and, if you want Slack alerts, create the Slack app from `deploy/slack-manifest.yaml` | A day to more than a week for eBay; minutes to days for Slack, depending on your workspace's app approval | |
| 2. Put `CRUSOE_API_KEY` in `.env`, then `eft doctor crusoe` | Minutes | Photos and descriptions become draft watches |
| 3. Deploy the deletion-notice responder in `deploy/worker/` | 20 minutes | You have the HTTPS endpoint eBay asks for |
| 4. When eBay approves you: create a **Production** keyset, subscribe it to marketplace account deletion notices with the responder's URL, put `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` in `.env`, then `eft doctor ebay` | Minutes | Search, dry runs, activation |
| 5. Add a channel: `SLACK_BOT_TOKEN` and `SLACK_USER_ID`, or the two Pushover variables, then `eft doctor alerts` | Minutes | Alerts |
| 6. `eft agent install` | Minutes | Polling in the background |

A regular eBay buyer account is not enough for step 4. The keys come from the eBay Developers Program, and a production keyset stays disabled until the deletion-notice step is done.

## Keys

| Symptom | Cause | Fix |
|---|---|---|
| `crusoe is not configured: set CRUSOE_API_KEY` | The variable is empty or the `.env` file was not found | `cp env.example .env` and fill it in, or export the variable |
| `.env ... unreadable` in `eft status` | Claude Code's sandbox denies reads of `.env` files | Export the variables in your shell profile, or run `eft` from your own terminal. The poller is not affected |
| Crusoe returns 401 or 403 | Key wrong, expired, or created for another project | Create a new key in the Crusoe Console under Intelligence Foundry > Inference |
| Crusoe returns 404 `model_not_found` after you set `EFT_MODEL_TEXT` or `EFT_MODEL_VISION` | Model ids are case-sensitive | Copy the id exactly from `eft doctor crusoe`, or remove the override |
| eBay token request returns 401 | Sandbox keys used against production, or the other way round | Use the Production keyset, or set `EBAY_ENV=sandbox` |
| eBay returns 403 on search | The keyset is not enabled yet | Finish the marketplace account deletion step in the developer portal |

Variables set in your shell take priority over `.env`. A stale export in a shell profile is a common reason a new key in `.env` appears not to work.

## Searching and matching

| Symptom | Cause | Fix |
|---|---|---|
| A watch has been silent for weeks | Query too narrow, wrong category, or the ceiling is below what the item sells for | `eft dryrun ID` shows what the search returns and where listings drop out |
| `status` says a watch's query is too broad | A search window returned more than three pages | Add the style code, a colorway nickname or a `must_have` term |
| Lots of "maybe: size" | Sellers left the size out of the item specifics | Set `strictness=normal` if it is `strict`, or accept the digest |
| A good listing was rejected | A field contradicted the spec | `eft explain CODE` shows the stage and the quote. Fix the spec with `eft watch set`, or mark the field `optional` or `na` |
| Nothing arrives overnight | The poller stops while a laptop sleeps | Alerts arrive on wake. For continuous polling run `eft tick` from cron or a systemd timer on a machine that stays on |
| `daily cap reached` | 4,800 eBay calls used today | Fewer watches, or fewer marked hot. eBay raises limits through its Application Growth Check |

## State

Everything lives in one directory: `~/Library/Application Support/ebay-fashion-tracker/` on macOS, `~/.local/share/ebay-fashion-tracker/` elsewhere, or wherever `EFT_STATE_DIR` points. It holds `tracker.db` and your reference photos. `eft uninstall` removes it.

If Claude Code's sandbox cannot write there, either add the directory to the sandbox's allowed write paths or set `EFT_STATE_DIR` to a directory inside your project.

Hosts the tool contacts: `api.ebay.com`, `i.ebayimg.com` (listing photos returned by the API), `api.inference.crusoecloud.com`, `slack.com`, `api.pushover.net`, and your heartbeat URL if you set one.
