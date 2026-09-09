# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=11", "resvg-py>=0.2"]
# ///
"""Render the Developer Hub artwork from the read-only Crusoe Brand Kit.

From the repository root:

    uv run assets/generate-artwork.py --brand-kit /path/to/Crusoe-Brand-Kit

Outputs, written next to this script:

    hub-banner.png / hub-banner.gif             banner still and animation
    repo-overview.png / repo-overview.gif       navigation flowchart, still and animated
    repo-overview-dark.png / repo-overview-dark.gif
    product-line.png / product-line-dark.png    Crusoe product families diagram
    icons/tile-*.png                            link-row tiles used in Markdown
    icons/logo-*.png                            third-party marks on neutral tiles

Colors come from tokens/colors.json. Type is ABC Diatype Mono (Bold and
Regular) and Suisse Int'l, rasterized into the images; font files are never
copied into the hub. Glyphs are the kit's icon library, recolored only by
swapping their single fill value. Logos are the original lockups, unaltered.
Use --only banner|overview|product-line|icons to regenerate a subset.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import resvg_py
from PIL import Image, ImageColor, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent
MONO_DIR = "assets/fonts/DINAMO Order 2025-2799319/Diatype Mono"
SANS_DIR = "assets/fonts/desktop"
ICON_FILL = "#161616"  # native fill of every glyph in the kit's icon library

# Icon names are the kit's file names under assets/icons/, minus ".svg".
# Link-row tiles are written to icons/; black tile, Hi-Vis Yellow glyph.
LINK_TILES = {
    "globe": ("Globe", "black", "hi-vis-yellow"),
    "document": ("Document", "black", "hi-vis-yellow"),
    "dashboard": ("Dashboard", "black", "hi-vis-yellow"),
    "blog": ("Blog", "black", "hi-vis-yellow"),
}
# Lane glyphs drawn inside the flowchart cards only; glyph in the lane color.
LANE_TILES = {
    "book": ("Book", "black", "cloud-blue"),
    "code": ("Code", "black", "sage-green"),
    "integration": ("Integration", "black", "sunrise-gold"),
    "layers": ("Layers", "black", "hi-vis-yellow"),
}

@dataclass
class Kit:
    root: Path
    colors: dict

    def color(self, token: str) -> tuple[int, int, int]:
        return ImageColor.getrgb(self.colors[token])

    def font(self, family: str, size: int) -> ImageFont.FreeTypeFont:
        files = {
            "mono-bold": f"{MONO_DIR}/ABCDiatypeMono-Bold.otf",
            "mono": f"{MONO_DIR}/ABCDiatypeMono-Regular.otf",
            "sans": f"{SANS_DIR}/SuisseIntl-Regular.otf",
            "sans-medium": f"{SANS_DIR}/SuisseIntl-Medium.otf",
        }
        return ImageFont.truetype(str(self.root / files[family]), size)

    def icon(self, name: str, size: int, ink: str) -> Image.Image:
        """Rasterize a kit glyph with its single fill swapped for a palette color."""
        svg = (self.root / "assets/icons" / f"{name}.svg").read_text()
        svg = svg.replace('<path fill="#FFF" fill-opacity=".01" d="M0 0h16v16H0z"/>', "")
        assert ICON_FILL in svg, f"{name}.svg does not use the expected fill"
        svg = svg.replace(ICON_FILL, self.colors[ink])
        png = resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size)
        return Image.open(io.BytesIO(bytes(png))).convert("RGBA")

    def logo(self, variant: str, width: int) -> Image.Image:
        with Image.open(self.root / f"assets/logos/png/crusoe-lockup-{variant}.png") as original:
            logo = original.convert("RGBA")
        return logo.resize((width, round(width * logo.height / logo.width)), Image.Resampling.LANCZOS)


def load_kit(root: Path) -> Kit:
    tokens = json.loads((root / "tokens/colors.json").read_text())["defaults"]
    return Kit(root, tokens)


def fitted_text(draw, position, content, font, fill, *, box, anchor="lt"):
    """Draw text and fail loudly if it leaves the allowed box (left, top, right, bottom)."""
    bounds = draw.textbbox(position, content, font=font, anchor=anchor)
    if bounds[0] < box[0] or bounds[1] < box[1] or bounds[2] > box[2] or bounds[3] > box[3]:
        raise ValueError(f"Text exceeds its box {box}: {content!r} -> {bounds}")
    draw.text(position, content, font=font, fill=fill, anchor=anchor)


def tile(kit: Kit, size: int, background: str, ink: str, icon: str) -> Image.Image:
    image = Image.new("RGBA", (size, size), kit.color(background))
    glyph = kit.icon(icon, round(size * 0.58), ink)
    offset = (size - glyph.width) // 2
    image.alpha_composite(glyph, (offset, offset))
    return image


def save_gif(frames, durations, path, sample_indices=(0, -1)):
    """Quantize every frame against one shared palette and write a looping GIF."""
    width, height = frames[0].size
    sample = Image.new("RGB", (width, height * len(sample_indices)))
    for row, index in enumerate(sample_indices):
        sample.paste(frames[index], (0, row * height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True)
    print(f"{path.name}: {path.stat().st_size:,} bytes, {len(frames)} frames, {sum(durations) / 1000:.2f}s")


def point_on_path(path, progress):
    lengths = [math.dist(s, e) for s, e in zip(path, path[1:])]
    remaining = progress * sum(lengths)
    for s, e, length in zip(path, path[1:], lengths):
        if remaining <= length:
            t = remaining / length
            return (s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t)
        remaining -= length
    return path[-1]


def path_prefix(path, progress):
    """The traversed part of an orthogonal path, for drawing a lit trail."""
    lengths = [math.dist(s, e) for s, e in zip(path, path[1:])]
    remaining = progress * sum(lengths)
    points = [path[0]]
    for s, e, length in zip(path, path[1:], lengths):
        if remaining <= length:
            t = remaining / length
            points.append((s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t))
            return points
        points.append(e)
        remaining -= length
    return points


def pulse(canvas, center, kit, halo_color, ring_color, size=(11, 5)):
    x, y = center
    overlay = Image.new("RGBA", canvas.size)
    motion = ImageDraw.Draw(overlay)
    motion.ellipse((x - size[0], y - size[0], x + size[0], y + size[0]), fill=(*kit.color(halo_color), 110))
    motion.ellipse((x - size[1], y - size[1], x + size[1], y + size[1]), fill=kit.color("hi-vis-yellow"), outline=kit.color(ring_color))
    canvas.alpha_composite(overlay)


# ---------------------------------------------------------------- icons ---

# Third-party marks under assets/third-party/, placed on black tiles.
# "mono" SVGs are rendered in white; "color" files keep their own colors.
LOGO_TILES = {
    "deepseek": ("deepseek.svg", "mono"),
    "google": ("google.svg", "mono"),
    "meta": ("meta.svg", "mono"),
    "moonshot": ("moonshot-kimi.svg", "mono"),
    "nvidia": ("nvidia.svg", "mono"),
    "openai": ("openai.svg", "mono"),
    "qwen": ("qwen.svg", "mono"),
    "zai": ("zai.png", "color"),
}


def logo_tile(kit: Kit, slug: str, size: int = 96) -> Image.Image:
    import re

    filename, mode = LOGO_TILES[slug]
    source = ASSETS / "third-party" / filename
    inner = round(size * 0.62)
    if source.suffix == ".svg":
        svg = source.read_text()
        if slug == "openai":
            # Drop the rounded background plate so only the blossom remains.
            svg = re.sub(r'<path d="M1 578\.4[^"]*"[^>]*/>', "", svg, count=1)
        if mode == "mono":
            svg = re.sub(r'fill="(?:#fff|#ffffff|white)"', 'fill="#FFFFFF"', svg, flags=re.I)
            svg = re.sub(r'(<svg\b[^>]*?)(?<!fill=")>', r'\1 fill="#FFFFFF">', svg, count=1)
        glyph = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=inner, height=inner)))).convert("RGBA")
    else:
        glyph = Image.open(source).convert("RGBA")
        glyph.thumbnail((inner, inner), Image.Resampling.LANCZOS)
    image = Image.new("RGBA", (size, size), kit.color("black"))
    image.alpha_composite(glyph, ((size - glyph.width) // 2, (size - glyph.height) // 2))
    return image


def render_tiles(kit: Kit) -> None:
    out = ASSETS / "icons"
    out.mkdir(exist_ok=True)
    for stale in out.glob("tile-*.png"):
        if stale.stem[5:] not in LINK_TILES:
            stale.unlink()
    for stale in out.glob("logo-*.png"):
        if stale.stem[5:] not in LOGO_TILES:
            stale.unlink()
    for slug, (icon, background, ink) in LINK_TILES.items():
        path = out / f"tile-{slug}.png"
        tile(kit, 96, background, ink, icon).save(path, optimize=True)
        print(f"{path.relative_to(ASSETS)}: {path.stat().st_size:,} bytes")
    for slug in LOGO_TILES:
        path = out / f"logo-{slug}.png"
        logo_tile(kit, slug).save(path, optimize=True)
        print(f"{path.relative_to(ASSETS)}: {path.stat().st_size:,} bytes")


# --------------------------------------------------------------- banner ---

BANNER = (1280, 380)
TILE, DEPTH = 96, 14

# Black background with Hi-Vis Yellow linework; the slabs keep their brand colors.
THEME = {
    "background": "black", "panel": "black", "panel-line": "hi-vis-yellow", "logo": "white",
    "title": "alabaster", "tagline": "hi-vis-yellow", "body": "cement-gray",
    "line": "hi-vis-yellow", "face": "black", "top": "moss-green", "side": "moss-green",
    "glyph": "hi-vis-yellow", "caption": "cement-gray", "pulse-halo": "sage-green",
    # Slab front, top, and side colors per layer, top layer first. Text is black on these fronts.
    "slabs": (
        ("sunrise-gold", "sunrise-gold-light", "sunset-gold"),
        ("sage-green", "sage-green-light", "moss-green-light"),
        ("slate-blue", "cloud-blue", "ocean-teal"),
    ),
    "slab-text": "black", "slab-detail": "black",
}

STACK_LAYERS = (
    ("Intelligence Foundry", "inference and fine-tuning"),
    ("Managed clusters", "Kubernetes and Slurm"),
    ("GPU cloud", "NVIDIA and AMD"),
)


def banner_base(kit: Kit, theme: dict, body_line: str) -> Image.Image:
    c = kit.color
    base = Image.new("RGBA", BANNER, c(theme["background"]))
    draw = ImageDraw.Draw(base)
    draw.polygon(((790, 0), (1280, 0), (1280, 380), (700, 380)), fill=c(theme["panel"]))
    if theme.get("panel-line"):
        draw.line((790, 0, 700, 380), fill=c(theme["panel-line"]), width=2)
    left_box = (0, 0, 745, 380)
    # One title line: the authentic lockup (wordmark plus burst) followed by "Developer Hub".
    lockup = kit.logo(theme["logo"], 250)
    base.alpha_composite(lockup, (56, 98))
    fitted_text(draw, (56 + lockup.width + 22, 98 + lockup.height // 2 + 2), "Developer Hub", kit.font("mono-bold", 50), c(theme["title"]), box=left_box, anchor="lm")
    fitted_text(draw, (60, 196), "Train, fine-tune, and serve models on Crusoe Cloud.", kit.font("sans-medium", 26), c(theme["tagline"]), box=left_box)
    fitted_text(draw, (60, 240), body_line, kit.font("sans", 21), c(theme["body"]), box=left_box)
    draw.rectangle((60, 290, 120, 294), fill=c("hi-vis-yellow"))
    return base


def slab_faces(left, y, width, height, depth):
    """Top and side faces of an isometric slab whose front face is (left, y, left+width, y+height)."""
    top_face = ((left, y), (left + depth, y - depth), (left + width + depth, y - depth), (left + width, y))
    side_face = ((left + width, y), (left + width + depth, y - depth), (left + width + depth, y + height - depth), (left + width, y + height))
    return top_face, side_face


def draw_iso_tile(kit, canvas, draw, left, top, theme, glyph, lit, size=TILE, depth=DEPTH):
    c = kit.color
    top_face, side_face = slab_faces(left, top, size, size, depth)
    draw.polygon(top_face, fill=c(theme["top"]))
    draw.polygon(side_face, fill=c(theme["side"]))
    for face in (top_face, side_face):
        draw.line(face + (face[0],), fill=c(theme["line"]), width=2)
    draw.rectangle((left, top, left + size, top + size), fill=c(theme["face"]), outline=c(theme["line"]), width=2)
    if glyph is not None:
        canvas.alpha_composite(glyph, (left + (size - glyph.width) // 2, top + (size - glyph.height) // 2))
    if lit:
        draw.rectangle((left + size - 20, top + 8, left + size - 8, top + 20), fill=c("hi-vis-yellow"), outline=c(theme["line"]))


def draw_stack_layer(kit, draw, theme, left, y, width, height, depth, index, lit):
    """One isometric slab with centered text. A lit slab is drawn slightly larger about its center."""
    c = kit.color
    label, detail = STACK_LAYERS[index]
    front, top_color, side_color = theme["slabs"][index]
    if lit:
        scale = 1.07
        cx, cy = left + width / 2, y + height / 2
        width, height, depth = round(width * scale), round(height * scale), round(depth * scale)
        left, y = round(cx - width / 2), round(cy - height / 2)
    top_face, side_face = slab_faces(left, y, width, height, depth)
    draw.polygon(top_face, fill=c(top_color))
    draw.polygon(side_face, fill=c(side_color))
    for face in (top_face, side_face):
        draw.line(face + (face[0],), fill=c(theme["line"]), width=2)
    draw.rectangle((left, y, left + width, y + height), fill=c(front), outline=c(theme["line"]), width=2)
    center_x = left + width // 2
    box = (700, 0, 1280, 380)
    fitted_text(draw, (center_x, y + round(height * 0.36)), label, kit.font("mono", 16), c(theme["slab-text"]), box=box, anchor="mm")
    fitted_text(draw, (center_x, y + round(height * 0.72)), detail, kit.font("sans", 14), c(theme["slab-detail"]), box=box, anchor="mm")


STACK_GEOMETRY = dict(left=836, width=330, height=50, depth=18, top=300, gap=16, app_top=36)


def stack_frame(kit, theme, base, lit_layers=(), pulse_t=None, tokens=0, app_lit=False):
    c = kit.color
    g = STACK_GEOMETRY
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    center_x = g["left"] + g["width"] // 2
    stack_top = g["top"] - 2 * (g["height"] + g["gap"]) - g["depth"]
    app_bottom = g["app_top"] + 64
    draw.line((center_x, app_bottom, center_x, stack_top), fill=c(theme["line"]), width=2)
    # Bottom layer first so upper slabs overlap it correctly.
    for position in range(len(STACK_LAYERS)):
        index = len(STACK_LAYERS) - 1 - position
        y = g["top"] - position * (g["height"] + g["gap"])
        draw_stack_layer(kit, draw, theme, g["left"], y, g["width"], g["height"], g["depth"], index, position in lit_layers)
    # The app tile and its caption zoom together when lit, about the tile's center.
    scale = 1.1 if app_lit else 1.0
    size, depth = round(64 * scale), round(10 * scale)
    app_center = (center_x, g["app_top"] + 32)
    tile_left, tile_top = app_center[0] - size // 2, app_center[1] - size // 2
    draw_iso_tile(kit, canvas, draw, tile_left, tile_top, theme, kit.icon("Chat", round(34 * scale), theme["glyph"]), False, size=size, depth=depth)
    fitted_text(draw, (tile_left + size + depth + 12, app_center[1]), "your app", kit.font("mono", round(17 * scale)), c(theme["caption"]), box=(700, 0, 1280, 380), anchor="lm")
    for index in range(tokens):
        y = stack_top - 16 - index * 18
        if y > app_bottom + 4:
            draw.rectangle((center_x - 5, y, center_x + 5, y + 10), fill=c("hi-vis-yellow"), outline=c(theme["line"]))
    if pulse_t is not None:
        y = stack_top - (stack_top - app_bottom) * pulse_t
        pulse(canvas, (center_x, y), kit, theme["pulse-halo"], theme["line"])
    return canvas


def render_banner(kit: Kit) -> None:
    theme = THEME
    base = banner_base(kit, theme, "Use hosted models, or run your own stack on GPU infrastructure.")
    still = stack_frame(kit, theme, base, tokens=2, app_lit=True)
    frames, durations = [], []

    def add(frame, ms):
        frames.append(frame.convert("RGB"))
        durations.append(ms)

    add(stack_frame(kit, theme, base), 800)
    for count in range(1, 4):  # light the slabs from the GPU layer upward
        add(stack_frame(kit, theme, base, lit_layers=set(range(count))), 320)
    for step in range(1, 7):
        add(stack_frame(kit, theme, base, lit_layers={0, 1, 2}, pulse_t=step / 6), 70)
    add(stack_frame(kit, theme, base, lit_layers={0, 1, 2}, app_lit=True), 300)
    for count in range(1, 3):
        add(stack_frame(kit, theme, base, lit_layers={0, 1, 2}, tokens=count, app_lit=True), 220)
    add(still, 1800)
    png = ASSETS / "hub-banner.png"
    still.convert("RGB").save(png, optimize=True)
    print(f"{png.name}: {png.stat().st_size:,} bytes, {BANNER[0]} x {BANNER[1]}")
    save_gif(frames, durations, ASSETS / "hub-banner.gif", sample_indices=(0, len(frames) // 2, -1))


# ------------------------------------------------------------- overview ---
# Left-to-right flowchart: a start node branches to four developer goals,
# each pointing at a section card that lists its subdirectories. The
# animation lights each route in turn and makes the card's bullets glow.

LANES = (
    {
        "goal": ("Learn the", "concepts"),
        "section": "foundations/",
        "tile": "book",
        "summary": "Concept lessons and teaching notebooks",
        "items": (
            ("gpu-engineering/", "Triton kernels, benchmarks, kernel inspection"),
            ("post-training/", "Quantization, LoRA, and QLoRA from scratch"),
            ("ai-eng/", "RAG, tool calling, agents, evaluation"),
            ("cloud/", "Storage, customer-managed keys, Slurm"),
        ),
        "background": "cloud-blue-light",
        "border": "slate-blue",
    },
    {
        "goal": ("Complete", "a task"),
        "section": "examples/",
        "tile": "code",
        "summary": "Runnable, task-oriented, with setup and cleanup",
        "items": (
            ("infrastructure/", "Provision compute, storage, and clusters"),
            ("training/", "Prepare data, fine-tune, deploy, evaluate"),
            ("inference/", "Call models and build serving workflows"),
            ("workshop/", "Guided sessions from conferences and events"),
            ("cookbooks/", "Short recipes with one concrete result"),
        ),
        "background": "sage-green-light",
        "border": "sage-green",
    },
    {
        "goal": ("Connect", "your tools"),
        "section": "integrations/",
        "tile": "integration",
        "summary": "Packages, gateways, editors, and search providers",
        "columns": (
            (("langchain/", "ChatCrusoe package"), ("litellm/", "Gateway"), ("mlflow/", "Deployment plugin"), ("google-adk/", "Tool-calling agent"), ("postman/", "API collection")),
            (("huggingface/", "Spaces demos"), ("cursor/", "Editor setup"), ("zed/", "Editor setup"), ("tavily/", "Web search"), ("linkup/", "Web search")),
        ),
        "background": "sunrise-gold-light",
        "border": "sunset-gold",
    },
    {
        "goal": ("Deploy a", "workflow"),
        "section": "solutions-library/",
        "tile": "layers",
        "summary": "Workflows that combine clusters, storage, serving, and monitoring",
        "columns": (
            (("GPU clusters", "NCCL-tested VMs"), ("Shared storage", "Volume drivers"), ("Slurm images", "Custom images")),
            (("Model serving", "KServe"), ("Pretraining", "TorchTitan"), ("Monitoring", "Grafana")),
        ),
        "column_gap": 410,
        "background": "cement-gray",
        "border": "stone-grey",
    },
)

OVERVIEW_WIDTH = 1400
CARD_LEFT, CARD_RIGHT = 590, 1370
ROW = 30
TRUNK_X = 262


def lane_entries(lane):
    """Flatten a lane's items or columns into (column, row, directory, note)."""
    if "columns" in lane:
        return [(col, row, d, n) for col, items in enumerate(lane["columns"]) for row, (d, n) in enumerate(items)]
    return [(0, row, d, n) for row, (d, n) in enumerate(lane["items"])]


def lane_rows(lane):
    return max(len(col) for col in lane["columns"]) if "columns" in lane else len(lane["items"])


class Overview:
    def __init__(self, kit: Kit, dark: bool):
        self.kit, self.dark = kit, dark
        c = kit.color
        self.background = c("moss-green") if dark else c("alabaster")
        self.ink = c("cement-gray") if dark else c("black")
        self.line = c("cement-gray") if dark else c("ocean-teal")
        self.heights = [112 + ROW * lane_rows(lane) for lane in LANES]
        gap, top0 = 28, 30
        self.tops = [top0 + sum(self.heights[:i]) + gap * i for i in range(len(LANES))]
        self.centers = [top + height // 2 for top, height in zip(self.tops, self.heights)]
        self.size = (OVERVIEW_WIDTH, self.tops[-1] + self.heights[-1] + 30)
        self.mid = (self.centers[0] + self.centers[-1]) // 2
        self.tiles = {lane["tile"]: tile(kit, 44, *LANE_TILES[lane["tile"]][1:], LANE_TILES[lane["tile"]][0]) for lane in LANES}

    def path_to_goal(self, index):
        y = self.centers[index]
        return ((230, self.mid), (TRUNK_X, self.mid), (TRUNK_X, y), (300, y))

    def path_to_card(self, index):
        y = self.centers[index]
        return ((510, y), (CARD_LEFT - 4, y))

    def marker_position(self, lane, col, row):
        top = self.tops[LANES.index(lane)]
        x = CARD_LEFT + 22 + col * lane.get("column_gap", 380)
        y = top + 100 + row * ROW + 6
        return x, y

    def base(self) -> Image.Image:
        kit, c = self.kit, self.kit.color
        canvas = Image.new("RGBA", self.size, self.background)
        draw = ImageDraw.Draw(canvas)
        mono_bold, mono, sans, sans_medium = kit.font("mono-bold", 30), kit.font("mono", 20), kit.font("sans", 21), kit.font("sans-medium", 24)
        note_font = kit.font("sans", 19)

        def arrow(start, end):
            draw.line((*start, *end), fill=self.line, width=2)
            x, y = end
            draw.polygon(((x, y), (x - 10, y - 5), (x - 10, y + 5)), fill=self.line)

        draw.line((TRUNK_X, self.centers[0], TRUNK_X, self.centers[-1]), fill=self.line, width=2)
        draw.line((230, self.mid, TRUNK_X, self.mid), fill=self.line, width=2)
        start_box = (30, self.mid - 62, 230, self.mid + 62)
        draw.rectangle(start_box, fill=self.background, outline=self.ink, width=2)
        draw.rectangle((32, self.mid - 60, 228, self.mid - 54), fill=c("hi-vis-yellow"))
        fitted_text(draw, (130, self.mid - 30), "What do you", kit.font("mono", 22), self.ink, box=start_box, anchor="mt")
        fitted_text(draw, (130, self.mid + 4), "need today?", kit.font("mono", 22), self.ink, box=start_box, anchor="mt")

        for lane, top, height, center in zip(LANES, self.tops, self.heights, self.centers):
            arrow((TRUNK_X, center), (300, center))
            goal_box = (300, center - 42, 510, center + 42)
            draw.rectangle(goal_box, outline=c(lane["border"]) if not self.dark else self.line, width=2)
            for i, text in enumerate(lane["goal"]):
                fitted_text(draw, (405, center - 30 + i * 30), text, sans_medium, self.ink, box=goal_box, anchor="mt")
            arrow((510, center), (CARD_LEFT - 4, center))
            card = (CARD_LEFT, top, CARD_RIGHT, top + height)
            draw.rectangle(card, fill=c(lane["background"]), outline=c(lane["border"]), width=2)
            canvas.alpha_composite(self.tiles[lane["tile"]], (CARD_LEFT + 22, top + 20))
            fitted_text(draw, (CARD_LEFT + 84, top + 20), lane["section"], mono_bold, c("black"), box=card)
            fitted_text(draw, (CARD_LEFT + 84, top + 60), lane["summary"], sans, c("black"), box=card)
            entries = lane_entries(lane)
            # Notes in a column start after that column's widest directory name.
            widest = {}
            for col, _, directory, _ in entries:
                widest[col] = max(widest.get(col, 0), draw.textlength(directory, font=mono))
            for col, row, directory, note in entries:
                x, y = self.marker_position(lane, col, row)
                draw.rectangle((x, y, x + 12, y + 12), fill=c("ocean-teal"))
                fitted_text(draw, (x + 24, y - 6), directory, mono, c("black"), box=card)
                note_width = widest[col] if "columns" in lane else draw.textlength(directory, font=mono)
                fitted_text(draw, (x + 24 + note_width + 18, y - 5), note, note_font, c("moss-green"), box=card)
        return canvas

    def frame(self, base, lit_paths=(), pulse_at=None, glowing=(), goal_lit=None):
        kit, c = self.kit, self.kit.color
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        for points in lit_paths:
            if len(points) > 1:
                draw.line([tuple(p) for p in points], fill=c("hi-vis-yellow"), width=3)
        if goal_lit is not None:
            y = self.centers[goal_lit]
            draw.rectangle((300, y - 42, 510, y + 42), outline=c("hi-vis-yellow"), width=3)
        if glowing:
            overlay = Image.new("RGBA", self.size)
            glow = ImageDraw.Draw(overlay)
            for x, y in glowing:
                glow.ellipse((x - 10, y - 10, x + 22, y + 22), fill=(*c("hi-vis-yellow"), 90))
            canvas.alpha_composite(overlay)
            draw = ImageDraw.Draw(canvas)
            for x, y in glowing:
                draw.rectangle((x - 1, y - 1, x + 13, y + 13), fill=c("hi-vis-yellow"), outline=c("ocean-teal"), width=2)
        if pulse_at is not None:
            pulse(canvas, pulse_at, kit, "sage-green" if not self.dark else "cloud-blue", "ocean-teal" if not self.dark else "cement-gray")
        return canvas

    def animate(self, base):
        frames, durations = [], []

        def add(frame, ms):
            frames.append(frame.convert("RGB"))
            durations.append(ms)

        add(base, 900)
        for index, lane in enumerate(LANES):
            goal_path, card_path = self.path_to_goal(index), self.path_to_card(index)
            for step in range(1, 11):
                t = step / 10
                add(self.frame(base, lit_paths=[path_prefix(goal_path, t)], pulse_at=point_on_path(goal_path, t)), 55)
            add(self.frame(base, lit_paths=[goal_path], goal_lit=index), 220)
            for step in range(1, 6):
                t = step / 5
                add(self.frame(base, lit_paths=[goal_path, path_prefix(card_path, t)], goal_lit=index, pulse_at=point_on_path(card_path, t)), 55)
            entries = lane_entries(lane)
            entries.sort(key=lambda e: (e[1], e[0]))
            per_frame = 2 if "columns" in lane else 1
            lit = []
            for start in range(0, len(entries), per_frame):
                lit.extend(self.marker_position(lane, col, row) for col, row, _, _ in entries[start:start + per_frame])
                add(self.frame(base, lit_paths=[goal_path, card_path], goal_lit=index, glowing=lit), 150)
            add(self.frame(base, lit_paths=[goal_path, card_path], goal_lit=index, glowing=lit), 900)
        add(base, 1200)
        return frames, durations


def render_overview(kit: Kit, name: str) -> None:
    overview = Overview(kit, dark=name == "dark")
    base = overview.base()
    suffix = "" if name == "light" else "-dark"
    png = ASSETS / f"repo-overview{suffix}.png"
    base.convert("RGB").save(png, optimize=True)
    print(f"{png.name}: {png.stat().st_size:,} bytes, {overview.size[0]} x {overview.size[1]}")
    frames, durations = overview.animate(base)
    save_gif(frames, durations, ASSETS / f"repo-overview{suffix}.gif", sample_indices=(0, len(frames) // 3, -2))


# --------------------------------------------------------- product line ---

# An inverted tree: root, two family nodes, then one leaf per product.
# Indented leaves hang from the leaf above them.
PRODUCT_LINE = (
    {
        "family": "Infrastructure Cloud",
        "node": "slate-blue",
        "leaf": "cloud-blue-light",
        "border": "slate-blue",
        "left": 190,
        "items": ((0, "GPU and CPU virtual machines"), (0, "Crusoe Managed Kubernetes"), (0, "Crusoe Managed Slurm"), (0, "Storage and networking"), (0, "Command Center")),
    },
    {
        "family": "Intelligence Foundry",
        "node": "sunrise-gold",
        "leaf": "sunrise-gold-light",
        "border": "sunset-gold",
        "left": 810,
        "items": ((0, "Managed Inference"), (1, "Serverless Inference"), (1, "Self-Serve Deployments"), (1, "Tailored Deployments"), (0, "Serverless Fine-Tuning")),
    },
)


def render_product_line(kit: Kit, name: str) -> None:
    c = kit.color
    dark = name == "dark"
    background = c("moss-green") if dark else c("alabaster")
    ink = c("cement-gray") if dark else c("black")
    line = c("cement-gray") if dark else c("ocean-teal")
    width, node_width, node_height = 1400, 400, 62
    leaf_height, row_gap, first_row = 46, 60, 244
    size = (width, first_row + 5 * row_gap + 10)
    canvas = Image.new("RGBA", size, background)
    draw = ImageDraw.Draw(canvas)
    node_font, leaf_font = kit.font("mono-bold", 24), kit.font("mono", 19)

    def junction(x, y):
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=c("hi-vis-yellow"), outline=line)

    root = (560, 30, 840, 92)
    draw.rectangle(root, fill=c("ocean-teal"))
    draw.rectangle((560, 30, 840, 36), fill=c("hi-vis-yellow"))
    fitted_text(draw, (700, 64), "Crusoe Cloud", kit.font("mono-bold", 26), c("alabaster"), box=root, anchor="mm")
    centers = [family["left"] + node_width // 2 for family in PRODUCT_LINE]
    draw.line((700, 92, 700, 122), fill=line, width=2)
    draw.line((centers[0], 122, centers[1], 122), fill=line, width=2)
    for family, center in zip(PRODUCT_LINE, centers):
        left = family["left"]
        draw.line((center, 122, center, 150), fill=line, width=2)
        node = (left, 150, left + node_width, 150 + node_height)
        draw.rectangle(node, fill=c(family["node"]))
        fitted_text(draw, (center, 150 + node_height // 2 + 1), family["family"], node_font, c("black"), box=node, anchor="mm")
        spine_x = left + 30
        rows = family["items"]
        last_main = max(i for i, (indent, _) in enumerate(rows) if indent == 0)
        draw.line((spine_x, 150 + node_height, spine_x, first_row + last_main * row_gap + leaf_height // 2), fill=line, width=2)
        sub_spine_x = spine_x + 60
        sub_rows = [i for i, (indent, _) in enumerate(rows) if indent == 1]
        if sub_rows:
            parent_bottom = first_row + (sub_rows[0] - 1) * row_gap + leaf_height
            draw.line((sub_spine_x, parent_bottom, sub_spine_x, first_row + sub_rows[-1] * row_gap + leaf_height // 2), fill=line, width=2)
        for row, (indent, label) in enumerate(rows):
            x = spine_x + 30 + indent * 60
            y = first_row + row * row_gap
            cy = y + leaf_height // 2
            origin = sub_spine_x if indent else spine_x
            draw.line((origin, cy, x, cy), fill=line, width=2)
            junction(origin, cy)
            leaf = (x, y, left + node_width - 10, y + leaf_height)
            leaf_fill = c(family["leaf"]) if not indent else (c("alabaster") if not dark else c("cement-gray"))
            draw.rectangle(leaf, fill=leaf_fill, outline=c(family["border"]), width=2)
            fitted_text(draw, (x + 16, cy + 1), label, leaf_font, c("black"), box=leaf, anchor="lm")
    suffix = "" if name == "light" else "-dark"
    path = ASSETS / f"product-line{suffix}.png"
    canvas.convert("RGB").save(path, optimize=True)
    print(f"{path.name}: {path.stat().st_size:,} bytes, {size[0]} x {size[1]}")


# ----------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand-kit", required=True, type=Path, help="Path to the local Crusoe Brand Kit")
    parser.add_argument("--only", choices=("banner", "overview", "product-line", "icons"), help="Regenerate one group only")
    args = parser.parse_args()
    kit = load_kit(args.brand_kit)
    if args.only in (None, "icons"):
        render_tiles(kit)
    if args.only in (None, "product-line"):
        for theme in ("light", "dark"):
            render_product_line(kit, theme)
    if args.only in (None, "banner"):
        render_banner(kit)
    if args.only in (None, "overview"):
        for theme in ("light", "dark"):
            render_overview(kit, theme)


if __name__ == "__main__":
    main()
