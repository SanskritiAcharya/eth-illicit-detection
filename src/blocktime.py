"""Convert Ethereum block numbers into wall-clock time.

Transactions returned by the API carry `blockNumber` but no timestamp, and asking
for the nested `block { timestamp }` on every transaction is both expensive against
the complexity limit and prone to timing out.

Ethereum's block interval is not constant -- it drifted around 14-15 seconds under
proof-of-work and became a fixed 12 seconds after the Merge -- so a single "seconds
per block" constant would be wrong by weeks at the edges. Instead we fetch a sparse
set of anchor blocks once, cache them, and linearly interpolate between neighbours.
Within an anchor interval the block rate is near enough constant that the error is
minutes, which is far finer than any feature we compute from it.
"""

from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone
from pathlib import Path

from .blockscout import fetch_block_timestamp

ROOT = Path(__file__).resolve().parent.parent
ANCHOR_PATH = ROOT / "data" / "raw" / "block_anchors.json"

# One anchor roughly every 250k blocks from Ethereum's genesis era to the present.
ANCHOR_STEP = 250_000
FIRST_ANCHOR = 250_000
LAST_ANCHOR = 21_000_000


def _parse(timestamp: str) -> float:
    """Parse Blockscout's ISO timestamp into a Unix epoch value."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


def build_anchors(path: Path = ANCHOR_PATH, force: bool = False) -> dict[int, float]:
    """Fetch (and cache) the block-number to timestamp anchor points."""
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        return {int(k): float(v) for k, v in cached.items()}

    anchors: dict[int, float] = {}
    for number in range(FIRST_ANCHOR, LAST_ANCHOR + 1, ANCHOR_STEP):
        try:
            timestamp = fetch_block_timestamp(number)
        except Exception:
            continue
        if timestamp:
            anchors[number] = _parse(timestamp)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in sorted(anchors.items())}, indent=1))
    return anchors


class BlockClock:
    """Maps block numbers to timestamps by interpolating between anchors."""

    def __init__(self, anchors: dict[int, float] | None = None):
        anchors = anchors or build_anchors()
        self.blocks = sorted(anchors)
        self.times = [anchors[b] for b in self.blocks]
        if len(self.blocks) < 2:
            raise ValueError("need at least two anchor blocks to interpolate")

    def timestamp(self, block: int) -> float:
        """Return the approximate Unix time of a block.

        Blocks outside the anchor range are extrapolated from the nearest pair,
        which keeps the function total -- a missing edge case here would otherwise
        drop transactions from the very start or the very end of the sample.
        """
        index = bisect.bisect_left(self.blocks, block)
        if index == 0:
            lo, hi = 0, 1
        elif index >= len(self.blocks):
            lo, hi = len(self.blocks) - 2, len(self.blocks) - 1
        else:
            lo, hi = index - 1, index

        block_lo, block_hi = self.blocks[lo], self.blocks[hi]
        time_lo, time_hi = self.times[lo], self.times[hi]
        span = block_hi - block_lo
        if span == 0:
            return time_lo
        return time_lo + (block - block_lo) * (time_hi - time_lo) / span

    def days_between(self, block_a: int, block_b: int) -> float:
        """Elapsed days between two blocks (always non-negative)."""
        return abs(self.timestamp(block_b) - self.timestamp(block_a)) / 86_400

    def datetime(self, block: int) -> datetime:
        return datetime.fromtimestamp(self.timestamp(block), tz=timezone.utc)


if __name__ == "__main__":
    anchors = build_anchors(force=True)
    clock = BlockClock(anchors)
    print(f"cached {len(anchors)} anchors -> {ANCHOR_PATH}")
    for block in (3_000_000, 4_832_686, 6_988_615, 15_537_394):
        print(f"  block {block:>10,} -> {clock.datetime(block):%Y-%m-%d %H:%M} UTC")
