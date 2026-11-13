# What eBay's agreements require, and how this tool complies

This example reads eBay through its official API under two documents: the [eBay User Agreement](https://www.ebay.com/help/policies/member-behaviour-policies/user-agreement?id=4259) and the [API License Agreement](https://developer.ebay.com/join/api-license-agreement). Read both before you use it; they change, and you are the one bound by them. This page is a summary of how the code is built, not legal advice.

| Requirement | How the tool handles it |
|---|---|
| No automated access to eBay's sites, which the User Agreement extends to "LLM-driven bots" | Listings are read only through `api.ebay.com`. `eft_core/http.py` refuses every request to an eBay site or short-link host, redirects included. Links are parsed as text. `SKILL.md` forbids Claude from fetching eBay pages |
| No order flow without human review | The tool never bids, makes offers or buys. It reports price and bid state and links to the listing |
| Listing data shown to a user must be fresh (six hours) | Every card carries an "as of" time. After six hours the poller strips price and bid fields from Slack messages. Phone pushes are sent with a six-hour expiry. Tracked listings are refreshed before that |
| Delete content that is no longer available | When a listing ends, its stored content and quoted evidence are removed and its messages are rewritten as "Ended". Rejected listings are stored as an id and a reason code only |
| No use of eBay content to train or evaluate models | Listings are sent to a model for inference only. The repository's test cases are hand-written, not eBay data |
| No price suggestions, models or derived statistics | The only price logic is a comparison with a ceiling the user typed. There is no deal score and no average |
| Respect call limits | A ledger counts calls by purpose and the client stops at 4,800 of the 5,000 daily calls |
| Marketplace account deletion notices | A production keyset must subscribe to them. `deploy/worker/` is a responder. The tool stores no seller or buyer identifiers, so there is nothing to delete on a notice |

Things that are yours to decide:

- **Where listing content goes.** Listing text and photos are sent to Crusoe Managed Inference to be judged. Under the API License Agreement a service provider's handling of eBay content is the developer's responsibility.
- **Whether your use fits the agreements at all.** They are written for applications; a personal tracker is a small case of one. If in doubt, ask eBay developer support.
