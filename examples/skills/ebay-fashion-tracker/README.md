# eBay Fashion Tracker

A [Claude Code skill](https://code.claude.com/docs/en/skills) that watches eBay for a specific fashion item and pings you when a matching listing appears. You describe the item with a photo, a link or a sentence; open models on [Crusoe Managed Inference](https://docs.crusoecloud.com/serverless-inference/index.html) turn that into a typed spec and judge every new listing against it; a background poller sends a Slack or phone alert with the price and bid state. You bid or buy on eBay yourself. The tool never does.

## What it shows

- **A skill whose model work runs on Crusoe.** Claude Code is the conversation. Vision and text models served by Crusoe read the photo, triage titles, verify listings and compare photos.
- **Models chosen at run time.** Each role picks a model from the live `/v1/models` catalog by capability, with fallbacks, and checks which model actually answered.
- **Model output as evidence, not as the decision.** Answers are JSON validated in code, quotes are checked against the listing text, and a fixed rule decides whether to alert.
- **A poller that costs no assistant tokens.** The same script runs on a schedule and calls eBay, Crusoe and Slack directly.
- **Working inside a marketplace's rules.** eBay is read only through its official API, listing data expires from alerts after six hours, and nothing is stored about sellers. See [references/ebay-rules.md](references/ebay-rules.md).

The design is in [ARCHITECTURE.md](ARCHITECTURE.md), including which parts version 0.1 implements.

## Status

Version 0.1. The code is covered by unit tests that run offline against fake API responses. It has **not yet been run against the live eBay, Crusoe, Slack or Pushover APIs**: `eft doctor` is the first live check, and you should expect it to surface differences the offline tests could not.

## Prerequisites

- Python 3.9 or newer. There are no packages to install.
- [Claude Code](https://code.claude.com/docs).
- A Crusoe API key, created in the [Crusoe Console](https://console.crusoecloud.com/) under Intelligence Foundry > Inference.
- An [eBay Developers Program](https://developer.ebay.com/) account and a **Production** keyset. Approval can take from a day to more than a week, and the keyset stays disabled until it is subscribed to marketplace account deletion notices.
- A Cloudflare account and Node.js, to deploy the small responder that eBay requires for those notices ([deploy/worker/](deploy/worker/)).
- One alert channel: a Slack app created from [deploy/slack-manifest.yaml](deploy/slack-manifest.yaml), or [Pushover](https://pushover.net/).

## Setup

```bash
cd examples/skills/ebay-fashion-tracker
cp env.example .env
chmod 600 .env
```

Fill in `.env`. Only `CRUSOE_API_KEY` is needed to start; the eBay and alert keys can follow.

```bash
python3 scripts/eft doctor env       # which keys are set, by name only
python3 scripts/eft doctor crusoe    # catalog, the model chosen for each role, a round trip
```

Make the skill available to Claude Code:

```bash
ln -s "$(pwd)" ~/.claude/skills/ebay-fashion-tracker
```

Then, as each piece becomes available:

| Step | Command to check it |
|---|---|
| Deploy the deletion-notice responder and enable the eBay keyset ([deploy/worker/README.md](deploy/worker/README.md)), then add `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` | `python3 scripts/eft doctor ebay` |
| Add the Slack or Pushover variables | `python3 scripts/eft doctor alerts` |
| Start the background poller (launchd on macOS; prints a cron line elsewhere) | `python3 scripts/eft agent install` |

Setup order, key problems and sandbox notes are in [references/troubleshooting.md](references/troubleshooting.md).

## Run

In Claude Code, talk to it:

- "Track these on eBay, size US 10, under $180" with a photo dragged in
- "Watch for this jacket in a medium" with an eBay or retailer link
- "What am I tracking?" / "Anything new?"
- "A17 isn't a match, it's the kids' version"
- "Why did A17 match?"

Or drive the engine yourself:

| Command | What it does |
|---|---|
| `eft add --image PATH --note "US 10, under 180"` | Drafts a watch from a photo. Also `--text` and `--url` (eBay links are parsed, never fetched) |
| `eft watch list`, `show`, `set`, `activate`, `pause`, `remove`, `hot` | Manage watches. A watch activates only once its spec is complete |
| `eft dryrun ID` | One search and its decisions, nothing kept |
| `eft tick` | One poll: search, judge, refresh tracked listings, send alerts |
| `eft inbox` | Alerts since you last looked |
| `eft explain CODE` | The stage-by-stage trace for one listing |
| `eft feedback CODE --reason "..."` | Marks a wrong match and proposes a change to the spec |
| `eft status`, `eft doctor` | Configuration, last poll, calls used today, what is still to set up |

Add `--json` to any command for machine-readable output.

## What to expect

`eft add` returns a draft spec and the questions the model could not answer from what you gave it, usually size, size system and price ceiling. After activation, each new listing ends in one of five states:

| Decision | Meaning |
|---|---|
| alert | Every required field was proven from the listing. You get a card |
| maybe | Nothing contradicts the spec but a required field is unproven. It goes in a daily digest that names the field |
| parked | It matches, but is over your ceiling or out of stock in your size. It is rechecked |
| reject | A field contradicts the spec, or the listing is a replica, a lot, customised or parts |
| pending | A model call failed. It is retried, then moved to maybe |

An alert card shows a short code, a proof line saying where each field was proven (for example `Size US 10 (item specifics)`), the price and shipping or the current bid, bid count and minimum next bid, the scheduled end, the time the data was read, the words "not authenticated", and the listing link from the API.

## Cost and limits

- **eBay calls.** The Browse API allows 5,000 calls a day per application. The tool stops at 4,800, which is enough for roughly 50 watches at a 10 to 40 minute cadence.
- **Model calls.** Estimated at a few US dollars a month at 25 watches. That is a planning estimate, not a measurement; `eft status` shows real call counts.
- **A laptop sleeps.** Under launchd the poller stops while the lid is closed. Nothing is lost, but alerts arrive on wake. Run `eft tick` from cron or a systemd timer on an always-on machine for continuous polling.
- **No authentication of items.** A photo match is evidence of model and colorway only.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests use the standard library only and make no network calls.

## Cleanup

```bash
python3 scripts/eft uninstall --yes     # stops the poller, deletes the database and state folder
rm ~/.claude/skills/ebay-fashion-tracker
```

Then delete the Worker (`npx wrangler delete` in `deploy/worker/`), remove the Slack app, and revoke the Crusoe and eBay keys if you no longer need them.

## References

- [Crusoe Managed Inference](https://docs.crusoecloud.com/serverless-inference/index.html) and its [available models](https://docs.crusoecloud.com/serverless-inference/available-models)
- [eBay Browse API](https://developer.ebay.com/api-docs/buy/browse/overview.html)
- [eBay API License Agreement](https://developer.ebay.com/join/api-license-agreement) and [User Agreement](https://www.ebay.com/help/policies/member-behaviour-policies/user-agreement?id=4259)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Skills](../)
