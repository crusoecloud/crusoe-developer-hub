# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=11"]
# ///
"""Render the tutorial figures for the GPU programming track from the Crusoe Brand Kit palette.

From this directory:

    uv run generate-figures.py --brand-kit /path/to/Crusoe-Brand-Kit

Every figure is a static PNG on a white background so it reads the same in both
GitHub themes. Type is ABC Diatype Mono and Suisse Int'l, rasterized at render
time; font files are never copied into the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
MONO_DIR = "assets/fonts/DINAMO Order 2025-2799319/Diatype Mono"
SANS_DIR = "assets/fonts/desktop"


class Kit:
    def __init__(self, root: Path):
        self.root = root
        self.colors = json.loads((root / "tokens/colors.json").read_text())["defaults"]

    def color(self, token):
        return ImageColor.getrgb(self.colors[token])

    def font(self, family, size):
        files = {
            "mono-bold": f"{MONO_DIR}/ABCDiatypeMono-Bold.otf",
            "mono": f"{MONO_DIR}/ABCDiatypeMono-Regular.otf",
            "sans": f"{SANS_DIR}/SuisseIntl-Regular.otf",
            "sans-medium": f"{SANS_DIR}/SuisseIntl-Medium.otf",
        }
        return ImageFont.truetype(str(self.root / files[family]), size)


class Figure:
    """Small drawing toolkit: sharp-cornered boxes, straight connectors, checked text."""

    def __init__(self, kit: Kit, size):
        self.kit = kit
        self.size = size
        self.image = Image.new("RGB", size, kit.color("alabaster"))
        self.draw = ImageDraw.Draw(self.image)
        self.line = kit.color("ocean-teal")

    def c(self, token):
        return self.kit.color(token)

    def text(self, xy, content, family="sans", size=18, fill="black", anchor="lt"):
        font = self.kit.font(family, size)
        bounds = self.draw.textbbox(xy, content, font=font, anchor=anchor)
        if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > self.size[0] or bounds[3] > self.size[1]:
            raise ValueError(f"text leaves the canvas: {content!r} {bounds}")
        self.draw.text(xy, content, font=font, fill=self.c(fill), anchor=anchor)

    def box(self, rect, fill, border=None, width=2):
        self.draw.rectangle(rect, fill=self.c(fill), outline=self.c(border) if border else None, width=width)

    def labelled_box(self, rect, title, subtitle=None, fill="cloud-blue-light", border="slate-blue", title_size=20, sub_size=15, ink="black", sub_ink="moss-green"):
        self.box(rect, fill, border)
        cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        if subtitle:
            self.text((cx, cy - 11), title, "mono", title_size, ink, "mm")
            self.text((cx, cy + 14), subtitle, "sans", sub_size, sub_ink, "mm")
        else:
            self.text((cx, cy), title, "mono", title_size, ink, "mm")

    def arrow(self, start, end, color=None, width=2, head=9):
        color = self.c(color) if color else self.line
        self.draw.line((*start, *end), fill=color, width=width)
        x, y = end
        dx, dy = end[0] - start[0], end[1] - start[1]
        if abs(dx) >= abs(dy):
            s = 1 if dx > 0 else -1
            self.draw.polygon(((x, y), (x - s * head, y - head // 2 + 1), (x - s * head, y + head // 2)), fill=color)
        else:
            s = 1 if dy > 0 else -1
            self.draw.polygon(((x, y), (x - head // 2, y - s * head), (x + head // 2, y - s * head)), fill=color)

    def line_to(self, start, end, color=None, width=2):
        self.draw.line((*start, *end), fill=self.c(color) if color else self.line, width=width)

    def caption(self, content, y=None):
        y = y if y is not None else self.size[1] - 30
        self.text((self.size[0] // 2, y), content, "sans", 16, "moss-green", "mm")

    def save(self, name):
        path = HERE / name
        self.image.save(path, optimize=True)
        print(f"{name}: {path.stat().st_size:,} bytes, {self.size[0]} x {self.size[1]}")


# ----------------------------------------------------------------- figures ---

def cpu_vs_gpu(kit):
    f = Figure(kit, (1400, 560))
    f.text((350, 36), "CPU: latency optimized", "mono-bold", 24, "black", "mm")
    f.text((1050, 36), "GPU: throughput optimized", "mono-bold", 24, "black", "mm")
    # CPU: a few large cores, big control and cache.
    left = 80
    f.labelled_box((left, 80, left + 540, 150), "Control", "branch prediction, out-of-order scheduling", "cement-gray", "stone-grey")
    for i in range(4):
        x = left + i * 135
        f.labelled_box((x, 170, x + 120, 290), "Core", "large, fast", "sunrise-gold-light", "sunset-gold")
    f.labelled_box((left, 310, left + 540, 380), "L1 / L2 / L3 cache", "megabytes, keeps one thread fed", "cloud-blue-light", "slate-blue")
    f.labelled_box((left, 400, left + 540, 470), "System memory (DRAM)", "hundreds of GB/s", "sage-green-light", "sage-green")
    # GPU: many small ALUs, small control and cache.
    right = 780
    f.labelled_box((right, 80, right + 540, 120), "Control", None, "cement-gray", "stone-grey", title_size=18)
    for row in range(4):
        for col in range(12):
            x = right + col * 45
            y = 135 + row * 40
            f.box((x, y, x + 38, y + 32), "sunrise-gold", "sunset-gold", 1)
    f.text((right + 270, 300), "thousands of simple arithmetic units, grouped into streaming multiprocessors", "sans", 15, "moss-green", "mm")
    f.labelled_box((right, 320, right + 540, 380), "L2 cache", "tens of MB, shared by every SM", "cloud-blue-light", "slate-blue")
    f.labelled_box((right, 400, right + 540, 470), "Global memory (HBM)", "terabytes per second", "sage-green-light", "sage-green")
    f.caption("A CPU spends its silicon on making one thread fast. A GPU spends it on running thousands of threads at once.", 520)
    f.save("cpu-vs-gpu.png")


def execution_model(kit):
    f = Figure(kit, (1400, 640))
    f.text((330, 36), "What you write (software)", "mono-bold", 22, "black", "mm")
    f.text((1070, 36), "What runs it (hardware)", "mono-bold", 22, "black", "mm")
    rows = (
        ("Thread", "one instruction stream on one element", "ALU", "one arithmetic unit inside an SM", "cloud-blue-light", "slate-blue"),
        ("Warp", "32 threads issued together in lockstep", "Scheduler", "issues one instruction to a warp", "cloud-blue-light", "slate-blue"),
        ("Block (Triton program)", "the tile one program instance owns", "Streaming Multiprocessor", "runs several blocks, has shared memory", "sage-green-light", "sage-green"),
        ("Grid", "every block launched by one kernel call", "GPU", "all SMs pull blocks until the grid is done", "sunrise-gold-light", "sunset-gold"),
    )
    for i, (name, note, hw, hw_note, fill, border) in enumerate(rows):
        y = 80 + i * 120
        f.labelled_box((80, y, 600, y + 90), name, note, fill, border)
        f.arrow((620, y + 45), (780, y + 45))
        f.text((700, y + 30), "runs on", "sans", 14, "moss-green", "mm")
        f.labelled_box((800, y, 1320, y + 90), hw, hw_note, fill, border)
        if i < len(rows) - 1:
            f.arrow((340, y + 92), (340, y + 116), width=2, head=7)
            f.text((352, y + 104), "grouped into", "sans", 13, "moss-green", "lm")
            f.arrow((1060, y + 92), (1060, y + 116), width=2, head=7)
            f.text((1072, y + 104), "contained in", "sans", 13, "moss-green", "lm")
    f.caption("Triton hands you the block. Threads and warps still exist underneath, but the compiler schedules them for you.", 610)
    f.save("execution-model.png")


def thread_hierarchy_frame(kit, block_chosen=True, show_panel=True, warps_shown=8, active_warp=2, active_thread=True, flash=False, legend=("thread", "warp", "block", "grid"), note=None):
    """One frame of the thread-hierarchy animation; the default arguments draw the finished still."""
    f = Figure(kit, (1400, 700))
    f.text((700, 36), "One kernel launch: grid of blocks, block of warps, warp of 32 threads", "mono-bold", 22, "black", "mm")
    gx, gy = 60, 90
    f.box((gx, gy, gx + 440, gy + 330), "sage-green-light", "sage-green")
    f.text((gx + 220, gy + 22), "Grid: 6 blocks (gridDim = 3 x 2)", "mono", 16, "black", "mm")
    for row in range(2):
        for col in range(3):
            x, y = gx + 24 + col * 136, gy + 50 + row * 132
            chosen = block_chosen and (row, col) == (0, 1)
            f.box((x, y, x + 112, y + 108), "sunrise-gold-light" if chosen else "cloud-blue-light", "sunset-gold" if chosen else "slate-blue")
            f.text((x + 56, y + 40), "Block", "mono", 15, "black", "mm")
            f.text((x + 56, y + 64), f"({col}, {row})", "mono", 15, "black", "mm")
    f.text((gx + 220, gy + 356), "blocks run independently and in any order", "sans", 14, "moss-green", "mm")
    f.text((gx + 220, gy + 378), "each one lands on whichever SM is free", "sans", 14, "moss-green", "mm")
    bx, by = 600, 90
    cell, gap = 18, 3
    width = 32 * (cell + gap) - gap
    if show_panel:
        # Guide lines from the chosen block to its expansion.
        f.line_to((gx + 24 + 136 + 112, gy + 50), (bx - 16, by), "sunset-gold")
        f.line_to((gx + 24 + 136 + 112, gy + 158), (bx - 16, by + 8 * (cell + gap) + 56), "sunset-gold")
        f.box((bx - 16, by, bx + width + 16, by + 8 * (cell + gap) + 56), "sunrise-gold-light", "sunset-gold")
        f.text((bx + width // 2, by + 22), "Block (1, 0): 256 threads = 8 warps x 32 threads", "mono", 16, "black", "mm")
        for warp in range(warps_shown):
            y = by + 46 + warp * (cell + gap)
            for t in range(32):
                x = bx + t * (cell + gap)
                fill, border, w = "cloud-blue-light", "slate-blue", 1
                if warp == active_warp:
                    fill, border = "slate-blue", "ocean-teal"
                    if flash:
                        border, w = "hi-vis-yellow", 2
                if warp == active_warp and t == 5 and active_thread:
                    fill, border = "hi-vis-yellow", "ocean-teal"
                f.box((x, y, x + cell, y + cell), fill, border, w)
            f.text((bx + width + 24, y + cell // 2), f"warp {warp}", "mono", 13, "black", "lm")
        if warps_shown == 8:
            f.text((bx + width // 2, by + 8 * (cell + gap) + 72), "threadIdx.x runs 0 to 255 across the block; warp k holds threads 32k to 32k + 31", "sans", 14, "moss-green", "mm")
    yl = 430
    items = (
        ("thread", "hi-vis-yellow", "ocean-teal", "one thread: one instruction stream, its own registers, one element of the tile"),
        ("warp", "slate-blue", "ocean-teal", "one warp: 32 threads that receive the same instruction at the same time (SIMT)"),
        ("block", "sunrise-gold-light", "sunset-gold", "one block: threads that share one SM's shared memory and can synchronize"),
        ("grid", "sage-green-light", "sage-green", "the grid: every block the launch created, covering the whole tensor"),
    )
    for i, (key, fill, border, label) in enumerate(items):
        y = yl + i * 40
        lit = key in legend
        f.box((600, y, 622, y + 22), fill if lit else "alabaster", border if lit else "stone-grey", 1)
        f.text((634, y + 11), label, "sans", 15, "black" if lit else "stone-grey", "lm")
    if note:
        f.text((700, 610), note, "sans-medium", 15, "ocean-teal", "mm")
    else:
        f.text((700, 610), "In Triton, BLOCK_SIZE sets how many elements a block covers and the compiler picks the thread count.", "sans", 15, "black", "mm")
    f.caption("A launch is a grid of blocks. Each block is a set of warps. Each warp is 32 threads that move in lockstep.", 665)
    return f


def thread_hierarchy(kit):
    still = thread_hierarchy_frame(kit)
    still.save("thread-hierarchy.png")
    frames, durations = [], []

    def add(frame, ms):
        frames.append(frame.image.convert("RGB"))
        durations.append(ms)

    add(thread_hierarchy_frame(kit, block_chosen=False, show_panel=False, legend=("grid",), note="Step 1: the launch creates a grid of blocks."), 1800)
    add(thread_hierarchy_frame(kit, show_panel=False, legend=("grid", "block"), note="Step 2: pick one block. It will run on one streaming multiprocessor."), 1500)
    add(thread_hierarchy_frame(kit, warps_shown=0, active_warp=None, active_thread=False, legend=("grid", "block"), note="Step 3: inside the block are 256 threads, organized as warps."), 1200)
    for w in range(1, 9):
        add(thread_hierarchy_frame(kit, warps_shown=w, active_warp=w - 1, active_thread=False, legend=("grid", "block"), note=f"Step 3: warp {w - 1} holds threads {32 * (w - 1)} to {32 * w - 1}."), 380)
    add(thread_hierarchy_frame(kit, active_thread=False, legend=("grid", "block", "warp"), note="Step 4: a warp is the unit the hardware schedules: 32 threads, one instruction at a time."), 1600)
    add(thread_hierarchy_frame(kit, legend=("grid", "block", "warp", "thread"), note="Step 5: each thread in the warp works on its own element with its own registers."), 1600)
    for _ in range(3):
        add(thread_hierarchy_frame(kit, flash=True, note="Step 6: SIMT. One instruction is issued and all 32 threads execute it together."), 350)
        add(thread_hierarchy_frame(kit, note="Step 6: SIMT. One instruction is issued and all 32 threads execute it together."), 350)
    add(still, 2600)
    sample = Image.new("RGB", (1400, 700 * 2))
    sample.paste(frames[0], (0, 0)); sample.paste(frames[-1], (0, 700))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    path = HERE / "thread-hierarchy.gif"
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True)
    print(f"thread-hierarchy.gif: {path.stat().st_size:,} bytes, {len(frames)} frames, {sum(durations) / 1000:.1f}s")


def cuda_vs_triton(kit):
    f = Figure(kit, (1400, 640))
    f.text((700, 36), "z = x + y for eight elements, split into two blocks of four", "mono-bold", 22, "black", "mm")
    pitch, cell_w, left = 70, 62, 700 - 4 * 70 + 4
    for row, name in enumerate(("x", "y")):
        y = 76 + row * 44
        f.text((left - 16, y + 18), f"{name} =", "mono", 18, "black", "rm")
        for i in range(8):
            x = left + i * pitch
            f.box((x, y, x + cell_w, y + 36), "cement-gray", "stone-grey", 1)
            f.text((x + cell_w // 2, y + 18), f"{name}{i}", "mono", 16, "black", "mm")
    divider = left + 4 * pitch - 4
    f.line_to((divider, 70), (divider, 190), "stone-grey")
    f.text((left + 2 * pitch - 4, 180), "Block 0: elements 0 to 3", "sans-medium", 15, "moss-green", "mm")
    f.text((left + 6 * pitch - 4, 180), "Block 1: elements 4 to 7", "sans-medium", 15, "moss-green", "mm")

    def panel(cx, title, subtitle):
        f.text((cx, 236), title, "mono-bold", 20, "black", "mm")
        f.text((cx, 262), subtitle, "sans", 15, "moss-green", "mm")

    # CUDA: one thread per element, eight scalar results.
    panel(350, "CUDA: thread level", "eight threads, each computes one scalar")
    t_w, t_pitch = 56, 62
    for b in range(2):
        base = 350 - 4 * t_pitch + 3 + b * (4 * t_pitch + 16) - 8
        for t in range(4):
            i = b * 4 + t
            x = base + t * t_pitch
            f.box((x, 290, x + t_w, 356), "cloud-blue-light", "slate-blue")
            f.text((x + t_w // 2, 306), f"T{t}", "mono", 15, "black", "mm")
            f.text((x + t_w // 2, 328), f"x{i}+y{i}", "mono", 13, "moss-green", "mm")
            f.arrow((x + t_w // 2, 358), (x + t_w // 2, 396), head=7)
            f.box((x, 398, x + t_w, 432), "sage-green-light", "sage-green")
            f.text((x + t_w // 2, 415), f"z{i}", "mono", 15, "black", "mm")
        f.text((base + 2 * t_pitch - 3, 456), f"Block {b}", "sans-medium", 14, "moss-green", "mm")
    f.text((350, 500), "You write the code for one thread and index it with threadIdx and blockIdx.", "sans", 15, "black", "mm")
    f.text((350, 524), "Threads in a block share memory and synchronize by hand.", "sans", 15, "black", "mm")

    # Triton: one program per block, one vector operation each.
    panel(1050, "Triton: block level", "two program instances, each computes one tile")
    p_w = 250
    for b in range(2):
        x = 1050 - p_w - 14 + b * (p_w + 28)
        f.box((x, 290, x + p_w, 356), "cloud-blue-light", "slate-blue")
        f.text((x + p_w // 2, 306), f"pid = {b}", "mono", 15, "black", "mm")
        lo, hi = b * 4, b * 4 + 4
        f.text((x + p_w // 2, 330), f"x[{lo}:{hi}] + y[{lo}:{hi}]", "mono", 14, "moss-green", "mm")
        f.arrow((x + p_w // 2, 358), (x + p_w // 2, 396), head=7)
        f.box((x, 398, x + p_w, 432), "sage-green-light", "sage-green")
        f.text((x + p_w // 2, 415), f"z[{lo}:{hi}]", "mono", 15, "black", "mm")
        f.text((x + p_w // 2, 456), f"Program {b}", "sans-medium", 14, "moss-green", "mm")
    f.text((1050, 500), "You write the code for one tile and index it with tl.program_id.", "sans", 15, "black", "mm")
    f.text((1050, 524), "The compiler decides how threads inside the tile share the work.", "sans", 15, "black", "mm")
    f.line_to((700, 230), (700, 540), "stone-grey")
    f.caption("Same input, same two-way split. CUDA assigns one thread per element; Triton assigns one program instance per block of elements.", 596)
    f.save("cuda-vs-triton.png")


def memory_hierarchy(kit):
    f = Figure(kit, (1400, 640))
    levels = (
        ("Registers", "per thread, private", "about 1 cycle", "tens of TB/s aggregate", 320, "sunrise-gold-light", "sunset-gold"),
        ("Shared memory / L1", "per SM, shared by a block", "about 30 cycles", "same on-chip SRAM since Volta", 460, "sunrise-gold-light", "sunset-gold"),
        ("L2 cache", "chip-wide, hardware managed", "about 200 cycles", "tens of MB", 600, "cloud-blue-light", "slate-blue"),
        ("Global memory (HBM)", "device-wide, where tensors live", "400 to 800 cycles", "tens of GB at a few TB/s", 740, "sage-green-light", "sage-green"),
        ("Host memory", "across PCIe or NVLink", "microseconds", "the other side of the bus", 880, "cement-gray", "stone-grey"),
    )
    top, center, cost_x = 70, 660, 1250
    for i, (name, scope, latency, note, width, fill, border) in enumerate(levels):
        y = top + i * 98
        left = center - width // 2
        f.box((left, y, left + width, y + 78), fill, border)
        f.text((center, y + 24), name, "mono-bold", 20, "black", "mm")
        f.text((center, y + 52), scope, "sans", 15, "moss-green", "mm")
        f.text((cost_x, y + 24), latency, "mono", 17, "black", "mm")
        f.text((cost_x, y + 52), note, "sans", 14, "moss-green", "mm")
    f.text((cost_x, 42), "access cost", "sans-medium", 15, "moss-green", "mm")
    f.arrow((120, top + 5 * 98 - 30), (120, top + 10))
    f.text((120, 42), "faster, smaller", "sans-medium", 15, "moss-green", "mm")
    f.arrow((60, top + 10), (60, top + 5 * 98 - 30))
    f.text((90, top + 5 * 98 - 8), "larger, slower", "sans-medium", 15, "moss-green", "mm")
    f.caption("A kernel that keeps its intermediate values in registers avoids a round trip that costs hundreds of cycles per element.", 610)
    f.save("memory-hierarchy.png")


def program_ids(kit):
    f = Figure(kit, (1400, 500))
    f.text((700, 36), "n_elements = 1000, BLOCK_SIZE = 256", "mono-bold", 22, "black", "mm")
    left, cell = 100, 300
    for pid in range(4):
        x = left + pid * cell
        fill = "cloud-blue-light" if pid < 3 else "sunrise-gold-light"
        border = "slate-blue" if pid < 3 else "sunset-gold"
        f.box((x, 90, x + cell - 6, 190), fill, border)
        f.text((x + cell // 2 - 3, 120), f"pid = {pid}", "mono-bold", 20, "black", "mm")
        lo = pid * 256
        f.text((x + cell // 2 - 3, 150), f"offsets = {lo} .. {lo + 255}", "mono", 15, "moss-green", "mm")
        f.text((x + cell // 2 - 3, 172), f"block_start = {pid} * 256", "mono", 14, "moss-green", "mm")
    # Tail: valid up to 999, masked after.
    x3 = left + 3 * cell
    valid_w = round((cell - 6) * (1000 - 768) / 256)
    f.box((x3, 200, x3 + valid_w, 236), "sage-green-light", "sage-green")
    f.box((x3 + valid_w, 200, x3 + cell - 6, 236), "cement-gray", "stone-grey")
    f.text((x3 + valid_w // 2, 218), "mask = True for 768 .. 999", "mono", 13, "black", "mm")
    f.text((x3 + (cell - 6) // 2, 252), "offsets 1000 .. 1023 are masked out of every load and store", "sans", 13, "moss-green", "mm")
    f.text((700, 280), "offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)", "mono", 20, "black", "mm")
    f.text((700, 316), "mask = offsets < n_elements", "mono", 20, "black", "mm")
    f.text((700, 366), "grid = (triton.cdiv(1000, 256),) = (4,)   four program instances, launched together", "sans", 17, "moss-green", "mm")
    f.caption("Each program computes the same offsets formula from its own pid. The mask keeps the last program inside the array.", 460)
    f.save("program-ids.png")


def compilation_pipeline(kit):
    f = Figure(kit, (1400, 420))
    stages = (
        ("Python", "@triton.jit source", "cement-gray", "stone-grey"),
        ("TTIR", "Triton IR: block-level ops, hardware agnostic", "cloud-blue-light", "slate-blue"),
        ("TTGIR", "Triton GPU IR: tiles mapped to warps, memory layout", "cloud-blue-light", "slate-blue"),
        ("LLVM IR", "per-thread operations", "sage-green-light", "sage-green"),
        ("PTX", "NVIDIA virtual assembly", "sage-green-light", "sage-green"),
        ("CUBIN", "machine code the GPU runs", "sunrise-gold-light", "sunset-gold"),
    )
    w, gap, left, top = 190, 40, 40, 90
    for i, (name, note, fill, border) in enumerate(stages):
        x = left + i * (w + gap)
        f.box((x, top, x + w, top + 110), fill, border)
        f.text((x + w // 2, top + 34), name, "mono-bold", 22, "black", "mm")
        for k, line in enumerate(wrap(note, 24)):
            f.text((x + w // 2, top + 64 + k * 18), line, "sans", 14, "moss-green", "mm")
        if i < len(stages) - 1:
            f.arrow((x + w + 4, top + 55), (x + w + gap - 4, top + 55))
    f.text((700, 44), "The first call compiles; every later call with the same constexpr values reuses the binary", "sans", 17, "black", "mm")
    for i, key in enumerate(("ttir", "ttgir", "llir", "ptx")):
        x = left + (i + 1) * (w + gap) + w // 2
        f.arrow((x, top + 112), (x, top + 150), head=7)
        f.text((x, top + 170), f'compiled.asm["{key}"]', "mono", 14, "ocean-teal", "mm")
    f.text((700, 300), "01_vector_add/v2_kernel_inspection.py writes these four stages to results/ so you can read them", "sans", 16, "moss-green", "mm")
    f.caption("Triton lowers a block-level Python function through its own IRs into the same kind of binary CUDA produces.", 380)
    f.save("compilation-pipeline.png")


def fusion(kit):
    f = Figure(kit, (1400, 620))
    f.text((350, 36), "Unfused: two kernels", "mono-bold", 22, "black", "mm")
    f.text((1050, 36), "Fused: one kernel", "mono-bold", 22, "black", "mm")

    def memory_bar(x, label):
        f.box((x, 70, x + 90, 500), "sage-green-light", "sage-green")
        f.text((x + 45, 90), "global", "mono", 14, "black", "mm")
        f.text((x + 45, 108), "memory", "mono", 14, "black", "mm")

    memory_bar(60, "")
    memory_bar(760, "")
    # Unfused traffic.
    f.labelled_box((260, 130, 640, 220), "silu_kernel", "temp = gate * sigmoid(gate)", "cloud-blue-light", "slate-blue")
    f.labelled_box((260, 330, 640, 420), "mul_kernel", "out = temp * value", "cloud-blue-light", "slate-blue")
    for y, label, rightward in ((150, "read gate", True), (200, "write temp", False), (350, "read temp", True), (380, "read value", True), (410, "write out", False)):
        if rightward:
            f.arrow((152, y), (258, y), "ocean-teal")
        else:
            f.arrow((258, y), (152, y), "sunset-gold")
        f.text((205, y - 12), label, "sans", 12, "moss-green", "mm")
    f.text((350, 530), "5 memory accesses per output element = 20 bytes in float32", "sans-medium", 16, "black", "mm")
    # Fused traffic.
    f.labelled_box((960, 210, 1340, 340), "swiglu_kernel", "out = (gate * sigmoid(gate)) * value, in registers", "cloud-blue-light", "slate-blue", sub_size=14)
    for y, label, rightward in ((240, "read gate", True), (275, "read value", True), (310, "write out", False)):
        if rightward:
            f.arrow((852, y), (958, y), "ocean-teal")
        else:
            f.arrow((958, y), (852, y), "sunset-gold")
        f.text((905, y - 12), label, "sans", 12, "moss-green", "mm")
    f.text((1050, 530), "3 memory accesses per output element = 12 bytes in float32", "sans-medium", 16, "black", "mm")
    f.caption("Same arithmetic, 40 percent less traffic. For a memory-bound kernel that is the whole speedup.", 585)
    f.save("fusion.png")


def rope_layout(kit):
    f = Figure(kit, (1400, 480))
    f.text((700, 36), "One program per sequence position m, feature_dim = 8, half_dim = 4", "mono-bold", 20, "black", "mm")
    left, cell = 200, 100
    f.text((left - 20, 118), "x[m]", "mono", 18, "black", "rm")
    for i in range(8):
        x = left + i * cell
        fill = "cloud-blue-light" if i < 4 else "sunrise-gold-light"
        border = "slate-blue" if i < 4 else "sunset-gold"
        f.box((x, 100, x + cell - 6, 136), fill, border)
        f.text((x + cell // 2 - 3, 118), f"x{i}", "mono", 16, "black", "mm")
    f.text((left + 2 * cell - 3, 80), "first half: offset m*8 + i", "sans", 14, "moss-green", "mm")
    f.text((left + 6 * cell - 3, 80), "second half: offset m*8 + i + 4", "sans", 14, "moss-green", "mm")
    for i in range(4):
        x0 = left + i * cell + cell // 2 - 3
        x1 = left + (i + 4) * cell + cell // 2 - 3
        f.line_to((x0, 138), (x0, 190 + i * 8), "ocean-teal")
        f.line_to((x0, 190 + i * 8), (x1, 190 + i * 8), "ocean-teal")
        f.arrow((x1, 190 + i * 8), (x1, 138), "ocean-teal", head=7)
    f.text((700, 250), "pairs (x_i, x_{i+4}) rotate together by angle m * omega_i", "sans", 16, "black", "mm")
    f.text((left - 20, 308), "cos[m]", "mono", 18, "black", "rm")
    for i in range(4):
        x = left + i * cell
        f.box((x, 290, x + cell - 6, 326), "sage-green-light", "sage-green")
        f.text((x + cell // 2 - 3, 308), f"c{i}", "mono", 16, "black", "mm")
    f.text((left + 2 * cell - 3, 350), "angle table stride is half_dim: offset m*4 + i", "sans", 14, "moss-green", "mm")
    f.text((900, 300), "out_i     = x_i * c_i - x_{i+4} * s_i", "mono", 16, "black", "lm")
    f.text((900, 330), "out_{i+4} = x_{i+4} * c_i + x_i * s_i", "mono", 16, "black", "lm")
    f.caption("The tensors have different row strides, so each load uses its own offset arithmetic. Getting the mapping right is the whole kernel.", 445)
    f.save("rope-layout.png")


def wrap(text, width):
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand-kit", required=True, type=Path)
    args = parser.parse_args()
    kit = Kit(args.brand_kit)
    for render in (cpu_vs_gpu, execution_model, thread_hierarchy, cuda_vs_triton, memory_hierarchy, program_ids, compilation_pipeline, fusion, rope_layout):
        render(kit)


if __name__ == "__main__":
    main()
