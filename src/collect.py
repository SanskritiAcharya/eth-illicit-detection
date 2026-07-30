"""Fetch raw transactions for every labelled address and cache them to disk.

This is the long pole of the project and the step the spec calls "freezing" the
data: we hit the API once, write the raw responses to data/raw/transactions.jsonl.gz,
and every later stage reads that file instead of the network. Re-running the
notebooks or retraining never touches Blockscout again.

The output is one JSON object per line:

    {"address": "0x...", "label": 1, "n_tx": 37, "transactions": [ ... ]}

Nothing is cleaned or aggregated here on purpose. Feature engineering happens in
src/features.py against the frozen file, so a change of mind about a feature costs
a few seconds instead of another hour of collection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .blockscout import fetch_transactions

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_PATH = RAW / "transactions.jsonl.gz"


def load_labels(path: Path) -> list[dict]:
    with path.open() as handle:
        return [
            {"address": row["address"].lower(), "label": int(row["label"])}
            for row in csv.DictReader(handle)
        ]


def already_collected(path: Path) -> set[str]:
    """Read back any addresses already in the cache so a re-run resumes.

    Collection takes the better part of an hour against a free public API. If it
    is interrupted we want to continue, not start over, so the output file is
    appended to and this set tells us what to skip.

    Only records that actually succeeded count as done. An address whose fetch
    failed -- because the rate limit was hit, say -- is deliberately left out, so
    a later run retries it instead of accepting a permanently empty row.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    try:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A partial final line from an interrupted run; ignore it.
                    continue
                if record.get("status") == "ok" and "address" in record:
                    done.add(record["address"])
    except (OSError, EOFError):
        # Truncated gzip member from a hard kill: keep whatever parsed.
        pass
    return done


def collect(
    labels: list[dict],
    out_path: Path,
    max_pages: int,
    workers: int,
    resume: bool = True,
) -> dict:
    """Fetch transactions for each address and append them to the cache.

    A small thread pool is used because the work is entirely network-bound, but
    the worker count barely matters: the real throughput ceiling is the server's
    budget of roughly 500 requests per fifteen minutes, and the shared RateLimiter
    in src/blockscout.py paces every worker against it. More threads would only
    mean more of them waiting.

    Failures are recorded rather than raised. An address that times out (quirk 5,
    the mega-address problem) simply contributes zero transactions and is counted
    in the summary, so we can report honestly how much of the sample we got.
    """
    done = already_collected(out_path) if resume else set()
    pending = [row for row in labels if row["address"] not in done]

    print(
        f"{len(labels)} labelled addresses, {len(done)} already cached, "
        f"{len(pending)} to fetch ({workers} workers, <={max_pages} pages each)",
        flush=True,
    )
    if not pending:
        return {"fetched": 0, "failed": 0, "skipped": len(done)}

    write_lock = threading.Lock()
    counters = {"fetched": 0, "failed": 0, "empty": 0, "tx_total": 0}
    started = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(out_path, "at")

    def worker(row: dict) -> None:
        try:
            transactions = fetch_transactions(row["address"], max_pages=max_pages)
            status = "ok"
        except Exception:
            # Quirk 5: mega-addresses and transient 5xx end up here. We keep the
            # address with an empty transaction list so the loss is visible in
            # the data rather than silently vanishing from the sample.
            transactions, status = [], "failed"

        record = {
            "address": row["address"],
            "label": row["label"],
            "n_tx": len(transactions),
            "status": status,
            "transactions": transactions,
        }

        with write_lock:
            handle.write(json.dumps(record) + "\n")
            if status == "failed":
                counters["failed"] += 1
            else:
                counters["fetched"] += 1
                counters["tx_total"] += len(transactions)
                if not transactions:
                    counters["empty"] += 1

            processed = counters["fetched"] + counters["failed"]
            # gzip buffers aggressively, so without an explicit flush an
            # interrupted run would leave an empty file and the resume logic
            # would have nothing to resume from.
            if processed % 25 == 0:
                handle.flush()
            if processed % 100 == 0:
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed else 0
                remaining = (len(pending) - processed) / rate / 60 if rate else 0
                print(
                    f"  {processed}/{len(pending)}  "
                    f"{counters['tx_total']} txs  "
                    f"{rate:.1f} addr/s  ~{remaining:.1f} min left",
                    flush=True,
                )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(worker, pending))
    finally:
        handle.close()

    counters["elapsed_s"] = round(time.time() - started, 1)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw transactions (freeze the data)")
    parser.add_argument("--labels", type=Path, default=RAW / "labels.csv")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="pages of 8 transactions per address; caps mega-addresses",
    )
    parser.add_argument("--workers", type=int, default=6, help="concurrent requests")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing cache")
    parser.add_argument("--limit", type=int, help="only collect the first N (for testing)")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]

    summary = collect(
        labels, args.out, args.max_pages, args.workers, resume=not args.no_resume
    )

    print("\nCollection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  cache: {args.out}")


if __name__ == "__main__":
    main()
