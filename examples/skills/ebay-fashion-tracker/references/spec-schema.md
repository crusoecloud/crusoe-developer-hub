# Item spec

A watch is a typed spec, not a keyword string. `eft watch show ID` prints it; `eft watch set ID KEY=VALUE` changes it.

## Fields

| Field | Example | Notes |
|---|---|---|
| `brand` | `Nike` | |
| `model` | `Dunk Low` | |
| `style_codes` | `DD1391-100` | Only from text read off the user's photo, typed by the user, or taken from a retailer page. Never from a model's memory |
| `colorway` | official `White/Black`, nicknames `Panda`, colors `white`, `black` | Nicknames widen the search |
| `size` | value `10`, system `US` | Systems: `US`, `UK`, `EU`, `JP`, `CM`, `alpha` (S, M, L), `waist_inseam` (`32x34`) |
| `department` | `men`, `women`, `unisex`, `kids` | With `kids_tier` `GS`, `PS` or `TD` when it applies |
| `conditions` | `NEW`, `USED` | Recorded; not yet enforced in version 0.1 |
| `price_ceiling` | amount `180`, currency `USD`, shipping included or not | A number the user typed. Listings above it are parked, not judged |
| `buying_options` | `FIXED_PRICE`, `AUCTION`, `BEST_OFFER` | All three by default |
| `must_have`, `must_not` | `OG box`; `Disrupt` | The user's own terms. `must_not` drops a listing on its title; `must_have` is recorded but not yet enforced |
| `visual_signature` | `white leather upper`, `black swoosh` | What the item looks like, in words. Recorded; a photo check against it is not in version 0.1 |
| `reference_images` | a path | The user's or a retailer's photo. eBay photos are never stored |
| `strictness` | `normal` or `strict` | See "How size is proven" |
| `requirements` | per field | See below |

## Required, optional, not applicable

Each of `brand`, `model`, `style_code`, `colorway`, `size` and `department` carries one of:

- `required`: must be proven from the listing before an alert fires.
- `optional`: a contradiction rejects the listing, but a missing value does not hold up an alert. Style code defaults to this, because sellers often leave it out.
- `na`: ignored. Use it for brand on an unbranded or vintage piece.

Defaults: a field the spec has a value for is `required`; a field it lacks is `na`.

```bash
eft watch set 3 require.brand=na require.style_code=required
```

## How size is proven

Size is the most common reason for a wrong alert, so the rule is narrow.

- A size counts as a **match** only when the listing states it in the watch's own size system: in the item specifics (for example `US Shoe Size: 10`), in the chosen variation of a multi-size listing, or read from the size tag in a photo.
- With `strictness=normal`, a size quoted from the title or description also counts. With `strict`, it does not.
- A size in another system is **never converted**. A UK 9 listing does not prove or disprove a US 10 watch; it stays unknown and the listing goes to the digest.
- Kids' sizes are their own thing: `7Y` is not `7`.
- A size in the title never rejects a multi-size listing; the variations are checked instead.

## How a decision is made

Models return `match`, `mismatch` or `unknown` per field with a quote from the listing. Quotes are checked to be real substrings of the listing text. Deterministic checks (size, style code, department) override the model when they have an answer. Then a fixed rule applies:

- any applicable field is a `mismatch`, or the listing is flagged as a replica, lot, custom or parts: **reject**
- matches, but over the ceiling or the size is out of stock: **parked** and rechecked
- every `required` field is a `match`: **alert**
- otherwise: **maybe**, naming the unproven fields

No model-reported confidence number is used anywhere.
