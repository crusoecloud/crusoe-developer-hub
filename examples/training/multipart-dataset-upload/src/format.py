"""Display helpers shared across the flow. `human_size` uses IEC binary labels
(KiB/MiB) since it divides by 1024, matching the product's 16 GiB / 128 MiB limits."""

from __future__ import annotations


def human_size(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val} B"
    value = float(bytes_val)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"


def human_duration(seconds: float) -> str:
    """A short H/M/S duration like '45s', '3m20s', or '1h05m'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
