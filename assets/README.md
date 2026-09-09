# Hub brand assets

Everything in this folder is either an unmodified platform logo or artwork generated from the Crusoe Brand Kit by [generate-artwork.py](generate-artwork.py). The brand kit itself is not part of the repository: it supplies the palette tokens, the ABC Diatype Mono and Suisse Int'l typefaces, the icon library, and the official lockups. Type is rasterized into the images; no font files are distributed here.

## Regenerate

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run assets/generate-artwork.py --brand-kit /absolute/path/to/Crusoe-Brand-Kit
```

The script declares its own dependencies (Pillow and resvg-py) and writes every output next to itself. Use `--only banner`, `--only overview`, `--only product-line`, or `--only icons` to regenerate one group. Text placement is checked at render time, so a wording change that no longer fits fails instead of clipping.

## Banner

[hub-banner.gif](hub-banner.gif) is the animation and [hub-banner.png](hub-banner.png) its still, both 1280 × 380 on a black background, so one version serves both GitHub themes. The left panel carries one title line made of the white lockup followed by "Developer Hub" in ABC Diatype Mono Bold, then "Train, fine-tune, and serve models on Crusoe Cloud." in Suisse Int'l Medium in Hi-Vis Yellow, a supporting line in Cement Gray, and a short Hi-Vis Yellow rule. A thin Hi-Vis Yellow diagonal separates the two panels. The right panel is a stack of three isometric slabs, the levels a developer can enter at: GPU cloud in Slate Blue, managed clusters in Sage Green, and Intelligence Foundry in Sunrise Gold, with Hi-Vis Yellow edges, feeding "your app" above them. Slabs use sharp corners and mono-thickness lines in the manner of the kit's technical illustrations. Slab labels are centered. In the animation each slab grows slightly in turn from the bottom up, a signal climbs to the app, the app tile and its caption grow together, and tokens appear on the connector.

The README serves the still through `prefers-reduced-motion`. Colors live in the `THEME` table in the generator.

## Product line diagram

[product-line.png](product-line.png) and [product-line-dark.png](product-line-dark.png) are an inverted tree: Crusoe Cloud at the root, the two families Infrastructure Cloud (Slate Blue) and Intelligence Foundry (Sunrise Gold) as nodes, and one leaf per product hanging from a spine, with the three Managed Inference options indented under their parent. Leaves carry names only; descriptions live in the README text and tables. Content is the `PRODUCT_LINE` table in the generator; keep labels aligned with the official product names on crusoe.ai and in the documentation.

## Navigation flowchart

[repo-overview.gif](repo-overview.gif) and [repo-overview-dark.gif](repo-overview-dark.gif) animate the four routes through the repository: a signal leaves the start node, follows the connector to a goal box, which lights up, continues to the section card, and the card's bullets glow one after another. Each route runs in turn; only the changed region is stored per frame, which keeps the file small. [repo-overview.png](repo-overview.png) and [repo-overview-dark.png](repo-overview-dark.png) are the stills for reduced-motion viewers.

The diagram is 1400 × 1102. Cards list subdirectories with a one-line description and carry no status markers; the section READMEs are the place for current detail. Cards use Cloud Blue Light, Sage Green Light, Sunrise Gold Light, and Cement Gray with black text in both themes; only the page background, connectors, and goal boxes change for dark mode. Contents live in the `LANES` table in the generator.

## Tiles

[icons/](icons/) holds 96 × 96 black tiles used inline in the hub README, matching the banner: four link-row tiles named `tile-*.png` with a Hi-Vis Yellow glyph, and eight model-provider marks in white on black named `logo-*.png`, with their originals and sources in [third-party/](third-party/). The section glyphs inside the flowchart cards are drawn at render time and are not written as files. See [icons/README.md](icons/README.md) for the full list.

## Badges

LinkedIn and Discord are local SVG badges that embed the platforms' own marks unchanged and make no remote requests:

| <div align="center">Badge</div> | <div align="center">Embedded mark and source</div> |
| --- | --- |
| [linkedin-badge.svg](linkedin-badge.svg) | White LinkedIn "in" logo from the [official LinkedIn downloads](https://brand.linkedin.com/downloads), `in-logo/InBug-White.png`, embedded unchanged |
| [discord-badge.svg](discord-badge.svg) | White Discord symbol from the [official Discord branding page](https://discord.com/branding), path embedded unchanged |

Platform colors identify the social destinations; Crusoe palette colors are reserved for repository navigation.

## Logo treatment

The Crusoe lockup appears only inside the generated banner, drawn from the brand kit at render time. Preserve its proportions, choose the variant that matches the background, and never recolor or redraw it. The kit does not publish numeric clear-space or minimum-size rules; keep generous separation as a production convention.

[Back to the hub](../README.md)
