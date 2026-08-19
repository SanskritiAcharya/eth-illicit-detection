# Detecting Illicit Ethereum Transactions with Graph-Derived Features

**DCS 404 — Machine Learning and Artificial Intelligence**
Sanskriti Acharya and Sweta Sharma
Week 8 final report

---

## 1. Problem definition

Cryptocurrency phishing works by volume. An attacker stands up a fake wallet
site, collects funds from whoever is fooled, and moves the proceeds through a
chain of addresses towards an exchange. Every one of those transfers is public
and permanently recorded, which sounds like it should make the problem easy. It
does not: Ethereum carries over a million transactions a day, and an
investigator cannot look at all of them.

What an exchange's compliance team or a blockchain forensics analyst actually
needs is a **short, prioritised list** — "these forty addresses are worth an
hour of your time" — rather than a blind search.

**The task.** Given an Ethereum address, predict whether it is involved in
phishing or scam activity.

**Track A — supervised binary classification.** Labels exist, and the target is a
binary class, so this is supervised classification rather than anomaly detection.
The project brief lists "anomaly detection in transactions" as a Track B example;
we deliberately did not take that route, because throwing away labels we have in
order to treat the problem as unsupervised would make the result both weaker and
harder to evaluate.

**Who it is for.** An analyst triaging addresses. That framing shapes the
application: it returns a probability and a review priority, not a bare verdict,
because the person using it decides what to investigate next.

**The research question.** The project is organised around one question:

> Can an address's network of connections reveal fraud that its own features cannot?

We answer it by training three tiers on an identical split and changing nothing
but the feature set.

### The metric, and why

**F1 on the illicit class**, reported alongside **PR-AUC**.

Accuracy is unusable here. 22.9% of our addresses are illicit, so a model that
answers "licit" every single time scores 77.1% accuracy across the dataset while
having learned nothing whatsoever — and on the held-out test split, where the
temporal ordering leaves only 15.5% positives, that same do-nothing model scores
**84.5%**. It is in our results table with an F1 of exactly 0.000, precisely so
this is impossible to gloss over.

The two mistakes also cost different amounts. A **missed scam** means fraud
continues and more victims pay. A **false alarm** costs an analyst a few minutes
of review. Recall therefore matters more than precision — but not infinitely,
since a model that flags everything is as useless as one that flags nothing. F1
is the harmonic mean of the two, so it only rises when both are respectable.
PR-AUC summarises the same trade-off across every threshold instead of only at
0.5, which is the right summary for an imbalanced problem.

---

## 2. Data

### Sources and licences

| | |
|---|---|
| **Labels** | MyEtherWallet `ethereum-lists` darklist — **MIT licence** |
| **Everything else** | Blockscout public API — free, no key, public on-chain data |
| **Size** | 2,852 addresses — 652 illicit, 2,200 licit (22.9% positive) |
| **Period** | 2017–2018 (blocks 2,912,407 – 6,988,615) |

The single most important property of this dataset is that we take **only labels**
from an external source. Every feature is computed by us from raw transaction
data that we fetched address by address. There is no pre-cleaned dataset download
anywhere in the project, and the address graph is built by us in NetworkX from
the transactions we collected.

### The illicit class

The MEW darklist is a community-maintained list of reported phishing and scam
addresses. It publishes 715 rows, which collapse to **652 distinct addresses** —
the same wallet is frequently reported under several different scam campaigns.
Deduplicating matters more than it sounds: left in, one address could appear in
both the training and the test split, quietly inflating every score we report.

### The licit class, and the trap we avoided

MEW publishes a matching "lightlist" of known-good addresses. We checked it: it
contains **two addresses**. It is unusable, so the licit class had to be sampled.

The obvious approach is to take ordinary addresses from recent blocks. That
approach silently destroys the project. **669 of the 715 darklist entries are
dated 2017 or 2018.** A licit class drawn from recent blocks would sit in a
completely disjoint block range, and any model could then separate the two
classes perfectly using block number alone. It would report superb metrics having
learned only that *old addresses are scams* — a fact about our sampling, not
about fraud.

So we sample licit addresses from **blocks in the same 2017–2018 window**. The EDA
notebook plots the two block-number distributions to confirm they overlap; if
they did not, nothing else in this report would mean anything.

Two further decisions follow the same logic:

- **Contracts are excluded from both classes.** The darklist is essentially all
  ordinary wallets, so allowing contracts into the licit class would hand the
  model "has contract code" as another giveaway unrelated to fraud.
- **Absolute block numbers are excluded from the features.** They are kept in the
  dataset because the temporal split needs them, but `src/features.py` defines
  exactly which columns a model may see, and `first_block` and `last_block` are
  not among them. Durations *derived* from blocks — lifetime, gaps between
  transactions — are kept, because a lifetime means the same thing whenever it
  happened.

### Collecting the data

Each address's transactions were fetched from Blockscout and frozen to
`data/raw/transactions.jsonl.gz`. Every later stage — features, graph, training,
notebooks — reads that file rather than the network, so the analysis is
reproducible and cannot drift.

Working against the live API produced seven undocumented behaviours, each of
which cost time and each of which is handled and commented in
`src/blockscout.py`:

| # | Behaviour | Handling |
|---|---|---|
| 1 | Python's default User-Agent gets `403` | Send a browser User-Agent (curl works without one; Python does not) |
| 2 | Query complexity capped at 100 | `first: 8` with 8 fields costs 96; `first: 9` costs 108 and is rejected |
| 3 | Requesting `cursor`/`endCursor` blows the same budget | Cursors are base64 `arrayconnection:<n>`; we construct them ourselves |
| 4 | `transactionsCount` is usually `null` | Never relied on; transactions counted by paging |
| 5 | Very large addresses hang the request | Pages per address capped; failures recorded, not retried forever |
| 6 | `gasUsed` returns `0` on many real transfers | Dropped in favour of `gas`, which is always populated |
| 7 | Undocumented limit of ~500 GraphQL requests/hour | Read `x-ratelimit-remaining`/`x-ratelimit-reset` and pace to fit |

Quirk 7 shaped the project. A short burst runs happily at four requests per
second, which is badly misleading — it is spending an hourly budget that then
leaves every subsequent request answered with `429`. At 1.4 requests per address,
collecting 2,852 addresses through GraphQL takes about **eight hours**.

We therefore run the bulk collection through Blockscout's REST endpoint, which
returns an address's transactions in a single response and completes the same
collection in about **twelve minutes**. This is worth being precise about, since
GraphQL was an explicit expectation for this project: GraphQL is what we explored
the chain with, what the application uses to score a live address, and what most
of `src/blockscout.py` implements. Both paths produce byte-identical records, and
`python -m src.collect --source graphql` still runs the entire collection through
GraphQL for anyone willing to wait out the rate limit.

---

## 3. EDA and preprocessing

Full detail is in `notebooks/01_eda.ipynb`. The decisions, and the reasoning
behind each:

| Decision | What we did | Why |
|---|---|---|
| Duplicate labels | 715 darklist rows → 652 unique addresses | The same wallet is reported under several campaigns; duplicates would span the train/test split. |
| Address case | Lower-cased everywhere | The darklist mixes checksummed and lower-case spellings of the same address. |
| Licit class source | Sampled from 2017–2018 blocks | 97% of the darklist is from that era; recent sampling would make block number a perfect separator. |
| Contracts | Excluded from both classes | Otherwise "has contract code" becomes a giveaway unrelated to fraud. |
| Empty addresses | Kept, features zeroed | Deleting them would change the class balance and hide a case the model must handle in practice. |
| `gasUsed` | Dropped in favour of `gas` | The API returns 0 for `gasUsed` on many ordinary transfers. |
| Block numbers | Kept for splitting, excluded from features | Absolute time invites memorising eras; derived durations are safe. |
| Correlated features | Kept | Forests are unaffected; the regression is standardised and L2-regularised. Interpretability beats tidiness here. |
| Scaling | `StandardScaler` inside the saved pipeline | The scaler is part of the model, so the app cannot preprocess differently than training did. |
| Skew | Left as-is, plotted on log axes | Trees ignore monotone transforms, and raw units stay explainable. |

### Features

29 behavioural features, all interpretable, computed per address: transaction
counts by direction, unique counterparties, value totals and means, net flow,
value concentration, gas price statistics, failure and self-transaction rates,
lifetime, transactions per active day, and timing regularity.

Two are worth singling out because they encode an actual fraud pattern:

- **`value_concentration`** — what fraction of everything an address sent left in
  its single largest transfer. A value near 1 is the signature of a wallet that
  gathers funds from many victims and then empties itself in one sweep.
- **`burstiness`** — the spread of gaps between transactions relative to their
  mean. Steady activity scores near 0; a wallet that fires many transactions
  within minutes and then goes silent scores high.

### Graph features

Every transaction is a directed edge between two addresses. The resulting graph
is much larger than the labelled set, because it also contains every counterparty
our labelled addresses touched — which is exactly the context these features are
meant to exploit. From it: degree, in/out degree, PageRank, clustering
coefficient, Louvain community size, and the neighbour risk ratio.

---

## 4. Modelling

### The split

Addresses are ordered by when they first appeared on chain; the earliest 70%
train, the most recent 30% test.

A random split would be easier and would score better, which is exactly the
problem with it. Scam operations run clusters of wallets that behave alike, and a
random split cheerfully puts one wallet from a cluster in training and its sibling
in test — the model then recognises the cluster rather than generalising. A
temporal split mimics the real task: learn from what is known, then score
addresses that came later.

### Preventing leakage

`neighbour_risk_ratio` asks "what fraction of my labelled neighbours are
illicit?". Computed over all labels, it leaks the answer outright: two scam
addresses that transact with each other would each reveal the other's label, and
the model would score brilliantly on a question it had been told the answer to.

`src/graph.py` therefore takes the visible labels as an explicit argument, and
training passes in **the training split's labels only**. Neighbours whose label
is unknown are excluded from the ratio rather than assumed licit, and
`n_labelled_neighbours` records how much evidence the ratio rests on, so the
model can learn to distrust it when that count is low. Unit tests in
`tests/test_pipeline.py` pin this behaviour down, including that an address can
never read its own label back through a self-loop.

### The four tiers

1. **Majority class.** Always predicts "licit". The trivial baseline the brief
   requires us to build, beat, and report.
2. **Behaviour only.** Logistic regression and random forest on the 29
   behavioural features.
3. **Behaviour + graph.** The same two models with the graph features added.
4. **Graph neural network.** A two-layer GraphSAGE (`src/gnn.py`) trained on the
   raw address graph itself — all 44k nodes, including the unlabelled one-hop
   neighbours. Where tier 3 hands a forest *our* summary of the network (degree,
   PageRank, neighbour risk), the GNN is given no graph features at all and has
   to learn what network context means by passing messages along the edges.
   Labelled nodes carry the 29 behaviour features; unlabelled neighbours carry
   only their degrees and a flag saying their behaviour was never collected.

Both classical classifiers are full scikit-learn pipelines with the scaler inside
them, and both use `class_weight="balanced"` so that missing the rarer illicit
class costs more. Hyperparameters are tuned by 5-fold cross-validation **on the
training split only**; the test split is touched exactly once, at scoring time.

The GNN follows the same rules translated to a neural setting: the identical
temporal split, a class-weighted loss over training nodes only, feature scaling
fit on training rows only, and early stopping against a validation slice carved
off the *newest end of the training window* — never the test split. It is
deliberately given no label-derived feature such as `neighbour_risk_ratio`;
feeding it our answer would make the comparison with tier 3 circular.

---

## 5. Results and evaluation

### The four tiers

All numbers are on the held-out temporal test split (856 addresses, 133 illicit).
Metrics are for the **illicit** class.

| Tier | Model | Accuracy | Precision | Recall | **F1** | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 — baseline | majority class | 0.845 | 0.000 | 0.000 | **0.000** | — | — |
| 2 — behaviour | logistic regression | 0.893 | 0.594 | 0.970 | **0.737** | 0.784 | 0.959 |
| 2 — behaviour | random forest | 0.959 | 0.806 | 0.970 | **0.881** | 0.952 | 0.988 |
| 3 — behaviour + graph | logistic regression | 0.930 | 0.695 | 0.977 | **0.812** | 0.799 | 0.967 |
| 3 — behaviour + graph | **random forest** | **0.963** | **0.822** | **0.970** | **0.890** | **0.969** | **0.993** |
| 4 — GNN | GraphSAGE | 0.939 | 0.731 | 0.962 | **0.831** | 0.884 | 0.972 |

The baseline is the row that justifies the whole metric choice: **84.5% accuracy
and an F1 of exactly zero**, from a model that has learned nothing at all. (It
scores higher than the dataset-wide 77.1% because the temporal split leaves the
test half with fewer positives — 133 of 856.) Any report quoting accuracy here
would be quoting a number a constant nearly matches.

Both real tiers beat it decisively. The shipped model is the tier-3 random
forest.

### Does the graph help?

| Comparison | Behaviour only | + graph | Change |
|---|---|---|---|
| Logistic regression, F1 | 0.737 | 0.812 | **+0.075** |
| Random forest, F1 | 0.881 | 0.890 | **+0.009** |
| Random forest, PR-AUC | 0.952 | 0.969 | **+0.017** |

The honest answer is **yes, but modestly, and it depends on the model**.

For logistic regression the gain is substantial: +0.075 F1, because a linear
model cannot construct anything like "how connected is this address" from the
behavioural columns, so the graph features tell it something genuinely new. For
the random forest the gain is small (+0.009 F1, +0.017 PR-AUC), because the
forest can already approximate much of that structure from counterparty counts.

Graph features account for roughly 9% of the forest's total feature importance,
with `out_degree`, `community_size`, and `pagerank` the most used. That is a real
contribution, but it is not the dominant one, and we would rather report +0.009
accurately than round it into a story about graphs transforming the problem.

The result should also be read against our graph's limits: it contains labelled
addresses and their immediate counterparties only, with at most 100 transactions
each. This measures the lift available from a **sparse** graph, not the ceiling.

### What the GNN adds to that answer

The GraphSAGE tier asks the same question from the other direction: instead of
us deciding what the network means and summarising it into eight columns, the
model reads the raw graph itself. It lands at **F1 0.831** — clearly above the
behaviour-only logistic regression (0.737) and roughly level with the tier-3
logistic regression (0.812), but below both random forests.

That ordering is informative rather than disappointing:

- **It confirms the graph carries real signal.** With 96% recall the GNN catches
  scams at the same rate as the forests, and it recovers most of the network
  signal without ever being shown a hand-crafted graph feature — including the
  withheld `neighbour_risk_ratio`, whose job it has to rediscover through
  message passing.
- **It loses on precision, and the reason is data size.** A GNN learns its own
  features, and 1,996 labelled training nodes on a sparse one-hop graph is a
  small corpus for that; the forest instead gets 37 features we engineered from
  domain knowledge, which is exactly the trade that favours classical models on
  small tabular-ish problems.
- **It says our hand-crafted features were not leaving much behind.** If degree,
  PageRank and neighbour risk had missed a large signal, a model free to learn
  arbitrary neighbourhood patterns should have found it and beaten tier 3. It
  did not — which is quiet evidence the feature engineering did its job.

The shipped model therefore stays the tier-3 random forest, with the GNN
reported as the strongest evidence we have that the answer to the research
question is a property of the data, not of one model family.

### Confusion matrix — shipped model

|  | predicted licit | predicted illicit |
|---|---|---|
| **actually licit** | 695 | 28 |
| **actually illicit** | 4 | 129 |

129 of 133 scam addresses caught, 4 missed, at the cost of 28 false alarms. For a
triage tool that is the right side of the trade: an analyst reviews 157 addresses
and finds 129 of the 133 real ones.

### Sensitivity: is this real, or did we build it?

This is the most important analysis in the project, and it nearly went unnoticed.

The licit class is sampled by reading the transactions of random blocks and
keeping the addresses found there. That is a **length-biased sample**: an address
that transacts often appears in more blocks, so it is more likely to be picked.
The illicit class carries no such bias. The symptom is stark — **61% of licit
addresses hit the 100-transaction collection cap, against 8% of illicit ones**.

So we tested whether the model had simply learned "busy means legitimate":

| Test | F1 | What it shows |
|---|---|---|
| `n_tx` alone | 0.836 | Transaction count by itself gets most of the way. The confound is real and large. |
| All 16 activity/volume features removed | 0.887 | Behaviour still classifies well without any measure of *how much* an address does. |
| **Activity-matched pairs** | **0.892** | Each illicit address paired with a licit address of near-identical transaction count (median 9 vs 10). |

The matched analysis is the decisive one. Within a matched pair, activity level
carries no information by construction, so anything the model achieves there comes
from *how* addresses behave rather than *how much*. It scores **0.892**, which is
not lower than the headline 0.890.

**The conclusion is that the confound is genuine but not load-bearing.** Raw
transaction count is contaminated by our sampling frame, and on its own it would
have produced a misleadingly good result. But the behavioural features carry
enough independent signal that removing the contamination entirely does not cost
any accuracy.

Reproduce with `python -m src.sensitivity`.

### Error analysis

The 4 missed scams and 28 false alarms have a shape. Missed addresses have very
little history — several have almost no transactions at all — so there is
genuinely nothing for the model to work from. This is a data-coverage limitation
rather than a modelling one.

The false alarms are more interesting, and they may not all be errors. Our licit
labels mean "never reported", not "verified clean". An address that behaves
exactly like a phishing wallet and has never been reported is precisely what an
undiscovered scam looks like. Our measured precision of 0.822 is therefore a
**lower bound**.

The predicted-probability histogram in `02_modeling.ipynb` shows a band of
addresses in the middle the model genuinely cannot separate. In deployment that
band is not a failure — it is the review queue, which is why the application
reports a probability and a priority rather than a verdict.

### The graph we built

| | |
|---|---|
| Nodes (addresses) | 44,321 |
| Edges (directed transfers) | 58,705 |
| Mean degree | 2.60 |
| Max degree | 330 |
| Connected components | 270 |

2,852 labelled addresses expand into a graph of 44,321 through their
counterparties — the context the graph features are computed over.

---

## 6. Application

The Streamlit application (`app/app.py`) loads the saved pipeline and offers
three tabs:

- **Look up an address** — paste any Ethereum address. Its transactions are
  fetched live through the GraphQL API, features are computed by the *same code*
  that built the training set, and the pipeline scores it. Returns a verdict, a
  probability, and a review priority band.
- **Explore features by hand** — set feature values directly, from presets such
  as "typical ordinary wallet" and "collect-and-sweep pattern", and watch the
  model respond without waiting on the network.
- **Model performance** — the three-tier comparison, feature importances, and
  graph statistics.

Two design decisions are worth stating.

**The app contains no preprocessing of its own.** It imports feature engineering
from `src/features.py` rather than reimplementing it. Copying preprocessing into
an application is the classic way for a deployed model to silently diverge from
the one that was evaluated, and the project brief names it explicitly.

**A live lookup cannot rebuild the training graph.** It can see the address and
its transactions, but not the labelled neighbourhood the graph features were
computed over. Those features are therefore set to their neutral values — the
same encoding used in training for an address with no labelled neighbours — and
the app says so on screen. A probability computed from partial evidence is
presented as exactly that.

Running it:

```bash
streamlit run app/app.py     # or: docker compose up --build
```

---

## 7. Limitations and reflection

**Labels are incomplete, and this is the deepest limitation.** The darklist is
community-reported, so an address scored LICIT has not been *cleared* — it has
simply never been reported. Some addresses in our licit sample are almost
certainly undiscovered scams, which means our measured precision is a *lower*
bound: some "false positives" may be correct detections of scams nobody has
reported yet.

**Coverage is capped.** Collection takes at most 100 transactions per address, so
features describe recent activity rather than a full history, and long-lived
addresses are under-described.

**The graph is sparse.** It contains labelled addresses and their immediate
counterparties only. Whatever lift the graph features provide is therefore what a
*sparse* graph gives, not a ceiling — a denser crawl would give them more to work
with.

**One chain, one period.** Trained on 2017–2018 Ethereum. Scam patterns evolve,
and nothing here is claimed to transfer to other chains or later periods.

### Ethical considerations

An address flagged by this model is a **candidate for review, not a suspect**. The
output ranks addresses for human attention; it is not evidence of wrongdoing and
must never be treated as such. Acting on a false positive could mean freezing an
innocent person's funds, which is why the application reports a probability and a
priority band rather than a verdict, and why its footer states plainly that a
LICIT score means "not reported", not "cleared".

There is also a base-rate problem worth naming. Illicit addresses are far rarer in
the wild than the 23% in our sample, so a model applied to all of Ethereum would
produce many more false positives per true detection than our test metrics
suggest. Any real deployment would need to be recalibrated against the true base
rate.

### What we would do differently

- **Crawl one hop further.** The graph features are the project's whole thesis
  and they are working with a thin graph. Collecting the counterparties'
  counterparties would test them properly.
- **Check the rate limit before designing around the API.** We measured
  throughput in a short burst, concluded four requests per second was
  sustainable, and built a plan on it. The limit is hourly, so the burst was
  spending a budget rather than measuring a rate — a full day of collection time
  was designed around a number that was never real.
- **Find more positives.** 652 addresses is workable but thin. OFAC's sanctioned
  address list would add a different flavour of illicit behaviour.

---

## 8. Use of AI assistance

An AI assistant (Claude) was used throughout this project as a reference and pair
programmer. Specifically: exploring the Blockscout GraphQL schema and diagnosing
its undocumented behaviour (the seven quirks in section 2), drafting and
reviewing the code in `src/`, and structuring the notebooks and written
documentation.

The modelling decisions are ours and we can explain each of them: the choice of
Track A, F1 on the illicit class as the metric and why accuracy is unusable, the
temporal split, the era-matched sampling strategy that keeps block number from
becoming a shortcut, and the leakage control on `neighbour_risk_ratio`. Every
line of code in this repository can be explained by the authors.

---

## References

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T.,
and Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics.* KDD Workshop on Anomaly
Detection in Finance.

Chen, W., Guo, X., Chen, Z., Zheng, Z., and Lu, Y. (2020). *Phishing Scam
Detection on Ethereum: Towards Financial Security for Blockchain Ecosystem.*
IJCAI.

Wu, J., Yuan, Q., Lin, D., You, W., Chen, W., Chen, C., and Zheng, Z. (2022).
*Who Are the Phishers? Phishing Scam Detection on Ethereum via Network
Embedding.* IEEE Transactions on Systems, Man, and Cybernetics.

MyEtherWallet (2026). *ethereum-lists*. https://github.com/MyEtherWallet/ethereum-lists (MIT licence).

Blockscout (2026). *Ethereum Explorer API*. https://eth.blockscout.com
