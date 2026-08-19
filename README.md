# Detecting Illicit Ethereum Transactions with Graph-Derived Features

**DCS 404 — Machine Learning and Artificial Intelligence**
Sanskriti Acharya and Sweta Sharma

A supervised binary classifier that scores an Ethereum address for signs of
phishing or scam activity, and a Streamlit application that puts it in front of a
user.

**Live demo:** <https://eth-illicit-detection.streamlit.app/>

The question the project is built around:

> **Can an address's network of connections reveal fraud that its own features cannot?**

To answer that, four model tiers are trained on an identical split — a trivial
baseline, behaviour-only features, behaviour plus graph-derived features, and a
graph neural network (GraphSAGE) that reads the raw address graph directly. The
gap between tiers 2 and 3 is the answer; the GNN checks it from the other
direction, with learned network representations instead of hand-crafted ones.

---

## What is actually ours

This matters more than it might sound, so it is stated first.

We take **only labels** from an external source: a community list of reported scam
addresses. We take **no features** from anyone.

- Every feature is computed by us from raw transaction data.
- The raw data is fetched by us from the Blockscout API, address by address.
- The address graph is built by us in NetworkX from the transactions we fetched.

There is no pre-cleaned dataset download anywhere in this project.

**A note on GraphQL.** GraphQL is the interface this project is built around: it
is what we explored the chain with, and what `src/blockscout.py` is mostly about.
The application scores a live address through REST first, falling back to
GraphQL if REST fails, because REST is the fast, reliable path and a live demo
should not be at the mercy of GraphQL's hourly rate limit. It is not what the bulk
collection runs on either, and the reason is measured rather than preferential — the
GraphQL endpoint allows **500 requests an hour** and caps a page at 8
transactions, which puts a 2,852-address collection at about **eight hours**. The
REST endpoint returns an address's transactions in one response, so the same
collection takes about **twelve minutes**. Both paths live in
`src/blockscout.py`, both emit byte-identical records, and `--source graphql`
still runs the whole collection through GraphQL for anyone willing to wait.

---

## Quick start

```bash
git clone https://github.com/SanskritiAcharya/eth-illicit-detection.git
cd eth-illicit-detection

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/app.py
```

The trained model is committed, so the app runs immediately without re-collecting
anything.

### With Docker

```bash
docker compose up --build
```

Then open <http://localhost:8501>.

### Rebuilding everything from scratch

```bash
python -m src.labels      # build the labelled address list
python -m src.blocktime   # cache block-number to timestamp anchors
python -m src.collect     # fetch raw transactions  (slow — see the note below)
python -m src.train       # engineer features, train tiers 1-3, save the pipeline
python -m src.gnn         # train the tier-4 GraphSAGE (needs requirements-gnn.txt)
```

`python -m src.train` runs end to end and writes `models/model.joblib`.

The GNN tier has its own dependency file so the deployed app stays light:

```bash
pip install -r requirements-gnn.txt   # torch + torch_geometric, CPU is fine
python -m src.gnn                     # trains in under a minute, appends to reports/metrics.csv
```

> **On collection time.** `src/collect.py` defaults to `--source rest` and takes
> about twelve minutes. `--source graphql` collects the same data through the
> GraphQL endpoint instead, which takes roughly eight hours because that endpoint
> allows only **500 requests per hour** — a limit that is undocumented and only
> announced by a `429` once the budget is gone. Either way the collector reads the
> rate-limit headers, paces itself, and is resumable: restart it and it picks up
> where it stopped.

---

## The data

| | |
|---|---|
| **Labels** | [MyEtherWallet `ethereum-lists`](https://github.com/MyEtherWallet/ethereum-lists) darklist — **MIT licence** |
| **Everything else** | [Blockscout](https://eth.blockscout.com) public API — free, no key, public on-chain data |
| **Size** | 2,852 addresses — 652 illicit, 2,200 licit (22.9% positive) |
| **Period** | 2017–2018 (blocks 2,912,407 – 6,988,615) |

`data/README.md` documents every file, its origin, and how to regenerate it.

### Why the licit addresses were sampled the way they were

MEW publishes a matching "lightlist" of known-good addresses. It contains **two
addresses**, so it is unusable, and the licit class had to be sampled.

The obvious approach — take addresses from recent blocks — quietly destroys the
project. 669 of the 715 darklist entries are dated 2017 or 2018, so a licit class
drawn from 2026 would sit in a completely different block range, and a model could
separate the classes perfectly using block number alone. It would score
beautifully having learned only that old addresses are scams.

So licit addresses are sampled from **blocks in the same 2017–2018 window**.
Contracts are excluded from both classes for the same reason: the darklist is
essentially all ordinary wallets, so contracts in the licit class would hand the
model "has contract code" as another giveaway unrelated to fraud.

`notebooks/01_eda.ipynb` plots the two block-number distributions to confirm they
overlap. If they did not, nothing else in the project would mean anything.

---

## Method

**1. Collect.** Fetch each labelled address's transactions from Blockscout and
freeze them to `data/raw/transactions.jsonl.gz`. Every later stage reads that file,
so the analysis is reproducible and never re-fetches.

**2. Engineer.** 29 interpretable per-address features — transaction counts,
unique counterparties, value totals and concentration, gas price statistics,
lifetime, timing regularity, failure rates. Each one is a plain count, ratio or
summary statistic that can be explained in a sentence.

**3. Build the graph.** Every transaction is a directed edge between two addresses.
From that graph: degree, in/out degree, PageRank, clustering coefficient, Louvain
community size, and the neighbour risk ratio.

**4. Model.** Four tiers. Logistic regression and random forest, tuned by 5-fold
cross-validation on the training split, plus a two-layer GraphSAGE trained on the
raw 44k-node address graph — no hand-crafted graph features, message passing has
to learn the network context itself.

**5. Ship.** A Streamlit app that scores a live address, and a Docker setup.

### Two correctness decisions worth reading

**The split is temporal, not random.** Addresses are ordered by first appearance;
the earliest 70% train, the latest 30% test. Scam operations run clusters of
wallets that behave alike, and a random split puts siblings from one cluster on
both sides — the model then recognises the cluster instead of generalising, and
the score flatters it.

**`neighbour_risk_ratio` sees training labels only.** This feature asks "what
fraction of my labelled neighbours are illicit?". Computed over all labels, it
leaks the answer outright: two scam addresses that transact with each other would
each reveal the other's label. `src/graph.py` takes the visible labels as an
explicit argument, and training passes in the training split alone.

### The metric

**F1 on the illicit class**, with PR-AUC alongside it.

About 23% of addresses are illicit, so a model that always answers "licit" scores
roughly 77% accuracy with an F1 of zero — it is in the results table for exactly
that reason. The two errors also cost different amounts: a missed scam lets fraud
continue, a false alarm costs an analyst a few minutes. F1 only rises when
precision and recall are both reasonable, and PR-AUC summarises the trade-off
across all thresholds instead of just at 0.5.

---

## Results

See `reports/metrics.csv` for the full table, and `notebooks/02_modeling.ipynb`
for the evaluation, error analysis, and threshold sweep. All four tiers are
reported together, as measured. Headline: the tier-3 random forest ships
(F1 0.890); the GraphSAGE lands at F1 0.831 with the same 96% recall, strong
evidence the graph signal is real — and that the hand-crafted graph features
were not leaving much behind.

---

## The application

```bash
streamlit run app/app.py
```

A deployed instance runs at <https://eth-illicit-detection.streamlit.app/>, so the
app can be tried without installing anything.

Three tabs:

- **Look up an address** — paste any Ethereum address. Its transactions are
  fetched live, features are computed with the same code that built the training
  set, and the saved pipeline scores it. Returns a verdict, a probability, and a
  review priority band.
- **Explore features by hand** — set feature values directly and watch the model
  respond, without waiting on the network.
- **Model performance** — the four-tier comparison, feature importances, and
  graph statistics.

A live lookup can see the address and its transactions but cannot rebuild the
training graph, so graph features are set to their neutral values and the app says
so on screen. A probability computed from partial evidence is presented as exactly
that.

---

## Repository layout

```
README.md  requirements.txt  Dockerfile  docker-compose.yml
data/
  README.md               sources, licences, how to regenerate
  raw/                    labels + API responses (transactions git-ignored)
  processed/              engineered features and the edge list
notebooks/
  01_eda.ipynb            exploration and preprocessing decisions
  02_modeling.ipynb       three tiers, evaluation, error analysis
src/
  blockscout.py           GraphQL client — API quirks and rate limiting
  labels.py               darklist positives + era-matched licit sampling
  blocktime.py            block number to timestamp interpolation
  collect.py              resumable collector; freezes the raw data
  features.py             per-address feature engineering
  graph.py                graph construction and network features
  train.py                tiers 1-3, temporal split, saves the pipeline
  gnn.py                  tier 4: GraphSAGE on the raw address graph
  predict.py              scoring interface used by the app
models/model.joblib       the full pipeline: scaler + classifier
models/gnn.pt             the trained GraphSAGE weights and config
app/app.py                Streamlit interface
.streamlit/config.toml    app theme, so every instance looks the same
reports/                  metrics table and figures
tests/                    unit tests
```

`src/` holds everything the application depends on. The app imports its
preprocessing from `src/features.py` rather than reimplementing it, so the numbers
on screen are produced by the same code that produced the training data.

`models/model.joblib` stores the **entire pipeline** — the `StandardScaler` and the
classifier together — because the scaler is part of the model, and a model saved
without it will silently mispredict.

---

## Working with the API

Seven quirks were measured against the live endpoint and are handled in
`src/blockscout.py`, each commented where it is handled. They are recorded here
because none are in the documentation and every one of them cost us time.

| # | Quirk | How it is handled |
|---|---|---|
| 1 | Python's default User-Agent gets `403` | Send a browser User-Agent (curl works without one; Python does not) |
| 2 | Query complexity is capped at 100 | `first: 8` with 8 fields costs 96; `first: 9` costs 108 and is rejected |
| 3 | Requesting `cursor`/`endCursor` blows the same budget | Cursors are base64 `arrayconnection:<n>`, so we construct them ourselves |
| 4 | `transactionsCount` is usually `null` | Never relied on; transactions are counted by paging |
| 5 | Very large addresses hang the request | Pages per address are capped; failures are recorded, not retried forever |
| 6 | `gasUsed` returns `0` on many real transfers | Dropped in favour of `gas`, which is always populated |
| 7 | Undocumented limit of ~500 requests / 15 min | Read `x-ratelimit-remaining` and `x-ratelimit-reset`; pace to fit |

Quirk 7 is the one that shapes the project. A short burst runs happily at four
requests per second, which is misleading — it is spending a budget that then
leaves every request answered with `429` for a quarter of an hour.

---

## Tests

```bash
python -m pytest tests/ -v
```

The tests cover the parts where a silent bug would be invisible in the output:
cursor construction, feature arithmetic on hand-built transactions, the leakage
guard in the graph features, and the fact that the saved pipeline carries its
scaler.

---

## Limitations

- **Labels are incomplete.** The darklist is community-reported, so an address
  scored LICIT has not been *cleared* — it has simply not been reported. Some
  addresses in the licit sample are very likely undiscovered scams.
- **100 transactions per address at most.** Collection caps each address at two
  pages, so features describe recent activity rather than a full history.
  Addresses with long histories are under-described.
- **The graph is sparse.** It contains labelled addresses and their immediate
  counterparties only, so the measured graph-feature lift is what a *sparse* graph
  gives, not a ceiling.
- **One chain, one period.** Trained on 2017–2018 Ethereum. Scam patterns change,
  and nothing here is claimed to transfer to other chains or later periods.
- **A triage aid, not evidence.** The output ranks addresses for human review. It
  is not proof of wrongdoing and should never be treated as such.

---

## Use of AI assistance

An AI assistant (Claude) was used during this project as a reference and pair
programmer: for exploring the Blockscout GraphQL schema and diagnosing its
undocumented behaviour, drafting and reviewing code in `src/`, and structuring the
notebooks and this README.

All modelling decisions — the choice of track, the metric and its justification,
the temporal split, the era-matched sampling strategy, and the leakage controls —
were made and are understood by the authors, who can explain every line of the
code and every decision behind it.

## Licence and attribution

Project code is available for academic use. Label data is from MyEtherWallet's
`ethereum-lists` under the MIT licence. Blockchain data is public and served by
Blockscout's free public API.
