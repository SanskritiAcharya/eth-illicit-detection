# Data

Nothing in `data/raw/` except this file and the label list is committed. The
transaction cache is regenerated from the API, and the instructions to do so are
below.

## What comes from where

| File | Origin | Committed? |
|---|---|---|
| `raw/darklist.json` | MyEtherWallet `ethereum-lists`, downloaded verbatim | yes |
| `raw/labels.csv` | Built by `src/labels.py`: darklist positives + sampled licit | yes |
| `raw/block_anchors.json` | Block-number to timestamp anchors from the API | yes |
| `raw/transactions.jsonl.gz` | Every address's raw transactions from the GraphQL API | **no** (git-ignored) |
| `processed/features_raw.csv` | Behavioural features, built by `src/features.py` | yes |
| `processed/features_full.csv` | Behavioural + graph features | yes |
| `processed/features_matched.csv` | Activity-matched licit/illicit pairs, built by `src/sensitivity.py` | yes |
| `processed/edges.csv` | Address-to-address edge list built from the transactions | yes |

## Sources and licences

**Labels — MyEtherWallet `ethereum-lists`.**
`https://github.com/MyEtherWallet/ethereum-lists`, MIT licence. We use
`src/addresses/addresses-darklist.json`: 715 rows of community-reported phishing
and scam addresses, which collapse to 652 distinct addresses. This is the *only*
external input to the project, and we take nothing from it but the address and
the fact that it was reported.

**Everything else — Blockscout public API.**
`https://eth.blockscout.com/api/v1/graphql`, free, no API key, no signup. The data
it serves is public Ethereum blockchain data. Every feature in this project is
derived by us from these raw responses.

## Reproducing the raw cache

```bash
python -m src.labels     # rebuild labels.csv       (a few minutes)
python -m src.blocktime  # rebuild block_anchors.json (about a minute)
python -m src.collect    # rebuild transactions.jsonl.gz
```

`src/collect.py` is resumable: it appends to the cache and skips addresses that
already succeeded, so an interrupted run can simply be restarted.

`src/collect.py` defaults to `--source rest` and takes about twelve minutes.
Passing `--source graphql` collects exactly the same records through the GraphQL
endpoint, which takes roughly eight hours: that endpoint allows only **500
requests per hour** — undocumented, and only revealed by a `429` once the budget
is gone — and caps a page at 8 transactions. The client reads the
`x-ratelimit-remaining` and `x-ratelimit-reset` headers and paces itself to fit,
so either run is unattended.

## Why the licit addresses are sampled the way they are

The obvious source of licit labels, MEW's matching `addresses-lightlist.json`,
contains only two addresses, so it is unusable.

Instead `src/labels.py` samples ordinary addresses from random blocks, and it
samples them from **2017 and 2018 specifically** — blocks 2,912,407 to 6,988,615.
That is the period 669 of the 715 darklist entries are dated to.

The reason is that the alternative quietly breaks the project. If the licit class
came from recent blocks, the two classes would occupy disjoint block ranges, and a
model could separate them perfectly from block number alone. It would score
beautifully while having learned only that old addresses are scams. Matching the
era removes that shortcut, so the model has to work from behaviour.

Contracts are excluded from both classes for the same reason: the darklist is
essentially all ordinary wallets, so allowing contracts into the licit class would
hand the model "has contract code" as another giveaway that says nothing about
fraud.

## Freezing

The spec asks for the data to be frozen, and it is: `src/collect.py` writes the
raw API responses once, and every later stage — feature engineering, the graph,
training, the notebooks — reads that file rather than the network. Re-running the
analysis does not re-fetch anything and cannot drift.
