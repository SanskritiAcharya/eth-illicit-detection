"""Build the labelled address list: illicit from MEW, licit sampled from the same era.

This is the only part of the project that uses an external source, and it uses it
for *labels only*. No feature ever comes from these files.

The illicit class is the MyEtherWallet `addresses-darklist.json` (MIT licensed):
phishing and scam addresses reported by the community.

The licit class is the part that needs care. MEW's matching `addresses-lightlist.json`
turned out to contain only two addresses, so it is unusable. We therefore sample
ordinary addresses ourselves, and we sample them from *the same period the darklist
covers*: 669 of the 715 darklist entries are dated 2017 or 2018.

If we sampled licit addresses from recent blocks instead, the two classes would sit
in disjoint block ranges and a model could separate them perfectly using block number
alone -- learning "old address = scam" rather than anything about behaviour. Matching
the era removes that shortcut and is what makes the reported scores meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

from .blockscout import USER_AGENT, fetch_address_meta, rest_block_transactions

DARKLIST_URL = (
    "https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/"
    "master/src/addresses/addresses-darklist.json"
)

# Block boundaries confirmed against the API: these are 2017-01-01 and 2019-01-01.
ERA_START_BLOCK = 2_912_407
ERA_END_BLOCK = 6_988_615

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def fetch_darklist() -> list[dict]:
    """Download the MEW darklist. Returns the raw records, duplicates included."""
    request = urllib.request.Request(DARKLIST_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def dedupe_addresses(records: list[dict]) -> list[dict]:
    """Collapse the darklist to one row per address.

    The published file has 715 rows but only 652 distinct addresses: the same
    wallet is often reported several times under different scam campaigns. Left
    unhandled, those 63 duplicates would put the same address in both the train
    and test split, which quietly inflates every score we report.

    Addresses are lower-cased because the file mixes EIP-55 checksummed and plain
    lower-case spellings of the same account.
    """
    seen: dict[str, dict] = {}
    for record in records:
        address = record["address"].strip().lower()
        if address not in seen:
            seen[address] = {
                "address": address,
                "label": 1,
                "source": "mew_darklist",
                "note": record.get("comment", "")[:120],
            }
    return list(seen.values())


def sample_licit_candidates(n_blocks: int, seed: int, exclude: set[str]) -> list[str]:
    """Draw candidate licit addresses from random blocks in the darklist's era.

    We read the transactions of randomly chosen blocks and keep the accounts that
    appear in them. This is a sampling frame only -- it tells us *which* addresses
    to look at, and every feature for them is then fetched through GraphQL.

    The GraphQL schema's Block type has no transaction list, so this listing has to
    come from Blockscout's REST endpoint. It is the single non-GraphQL call in the
    project and it contributes no feature values.
    """
    rng = random.Random(seed)
    blocks = rng.sample(range(ERA_START_BLOCK, ERA_END_BLOCK), n_blocks)
    candidates: list[str] = []
    seen: set[str] = set()

    for i, block in enumerate(blocks, 1):
        for tx in rest_block_transactions(block):
            for side in ("from", "to"):
                party = tx.get(side) or {}
                address = (party.get("hash") or "").lower()
                # Skip the darklist itself so a known scam wallet can never be
                # sampled as licit, and skip addresses we already hold.
                if address and address not in seen and address not in exclude:
                    seen.add(address)
                    candidates.append(address)
        if i % 20 == 0:
            print(f"  sampled {i}/{n_blocks} blocks -> {len(candidates)} candidates", flush=True)

    rng.shuffle(candidates)
    return candidates


def filter_to_accounts(candidates: list[str], target: int, batch: int = 20) -> list[dict]:
    """Keep ordinary user accounts, dropping smart contracts and empty accounts.

    Contracts are excluded on purpose. The darklist is essentially all
    externally-owned wallets, so if the licit class contained contracts then
    "has contract code" would be another giveaway that separates the classes
    without saying anything about fraud. Restricting both classes to ordinary
    accounts keeps the comparison honest.

    `contractCode` is non-null exactly for contracts. `nonce` counts the
    transactions an account has *sent*; requiring it to be non-null drops
    addresses that have only ever received funds and would yield almost no
    behaviour to learn from.
    """
    kept: list[dict] = []
    for start in range(0, len(candidates), batch):
        if len(kept) >= target:
            break
        chunk = candidates[start : start + batch]
        try:
            metas = fetch_address_meta(chunk)
        except Exception as exc:
            print(f"  meta batch failed ({exc}); skipping", flush=True)
            continue

        for meta in metas:
            if meta.get("contractCode") is not None:
                continue
            if meta.get("nonce") in (None, 0):
                continue
            kept.append(
                {
                    "address": meta["hash"].lower(),
                    "label": 0,
                    "source": "era_block_sample",
                    "note": "",
                }
            )
        if len(kept) % 200 < batch:
            print(f"  kept {len(kept)}/{target} licit accounts", flush=True)

    return kept[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labelled address list")
    parser.add_argument("--licit", type=int, default=2200, help="licit addresses to keep")
    parser.add_argument("--blocks", type=int, default=120, help="era blocks to sample")
    parser.add_argument("--seed", type=int, default=404, help="RNG seed (reproducibility)")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    print("Fetching MEW darklist...", flush=True)
    raw_records = fetch_darklist()
    (RAW / "darklist.json").write_text(json.dumps(raw_records, indent=1))
    illicit = dedupe_addresses(raw_records)
    print(f"  {len(raw_records)} rows -> {len(illicit)} unique illicit addresses", flush=True)

    print(f"Sampling licit candidates from {args.blocks} blocks in 2017-2018...", flush=True)
    exclude = {row["address"] for row in illicit}
    candidates = sample_licit_candidates(args.blocks, args.seed, exclude)
    print(f"  {len(candidates)} distinct candidate addresses", flush=True)

    print(f"Filtering to ordinary accounts (target {args.licit})...", flush=True)
    licit = filter_to_accounts(candidates, args.licit)
    print(f"  kept {len(licit)} licit accounts", flush=True)

    rows = illicit + licit
    out = RAW / "labels.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["address", "label", "source", "note"])
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(r["label"] for r in rows)
    print(
        f"\nWrote {out} -- {len(rows)} addresses, "
        f"{positives} illicit ({positives / len(rows):.1%} positive)"
    )


if __name__ == "__main__":
    main()
