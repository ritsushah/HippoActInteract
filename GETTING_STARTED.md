# Getting started with HippoActInteract

Welcome. This file is for someone who has **never** taken a cell-biology class, never heard of the Hippo pathway, and never trained a machine-learning model. You do not need any of that to follow along.

Read it from top to bottom. Every technical word is explained the first time it appears. By the end you should know what this project does, why it exists, what the numbers mean, and how to open the main results file.

If you only remember one sentence, remember this:

> The project is a carefully labeled **catalog** of possible connections between two groups of proteins. A computer model ranks the unlabeled ones. The catalog is the point. The ranks are a helper column, not a discovery of new biology.

---

## What is this project about?

Inside every cell, proteins do the work. They fold into shapes, stick to each other, and pass messages. Two of those messaging systems matter together:

- The **Hippo pathway** helps a tissue decide when to grow and when to stop.
- The **actin cytoskeleton** is the cell’s internal scaffolding — a mesh of protein filaments that gives the cell shape and lets it feel push and pull.

Scientists have long suspected these two systems talk to each other. The hard part is knowing **which proteins actually meet**. Public databases disagree. One database may list a pair as “associated.” Another may have a lab assay. A third may have nothing.

This project does three practical things:

1. **Draw a small neighborhood** of 359 proteins around Hippo (in humans) and a related yeast module called RAM/MOR, plus actin-related proteins.
2. **Score every Hippo × actin pair** in that neighborhood against three public catalogs (STRING, BioGRID, and IntAct) and tag each pair with an **evidence class**.
3. **Train a graph neural network** to rank pairs that STRING does not already list as high-confidence associations.

The result is an **evidence atlas**: a table of 160 protein pairs, each tagged with what is actually known (or not known) about it. The main file is `data/processed/hippo_actin_atlas.csv`.

The project does **not** prove that two proteins touch. A high computer score is a suggestion, not a lab result.

---

## Biological background

### Cells, proteins, and “pathways”

A **cell** is the basic unit of living tissue. Inside it, **proteins** are molecular machines made from chains of amino acids (think of amino acids as 20 kinds of beads on a string). The order of beads is the protein’s **sequence**. Sequence determines shape; shape determines what the protein can grab.

A **pathway** is not a hallway. It is a chain of proteins that pass a signal, often by chemically modifying one another. A typical pattern: protein A activates protein B, which then holds protein C out of the nucleus (the cell’s DNA compartment). When the hold is released, C enters the nucleus and turns genes on.

### Why actin matters

**Actin** is a protein that stacks into long filaments. Those filaments, plus motors that pull on them, are the **cytoskeleton** — the cell’s bones and cables. Cortical actin (the mesh just under the membrane) helps a dividing cell pinch in two, positions internal machinery, and transmits mechanical load into sticky contact points on the cell surface.

Actin is also a sensor. Stiffness of the surroundings, crowding, and which way the cell is polarized all feed through this mesh into signaling pathways.

### What the Hippo pathway does

**Hippo** is one of those pathways. In animals it helps control organ size and a behavior called **contact inhibition**: cells stop dividing when they become packed.

A simplified human version:

1. Kinases **MST1/2** (genes *STK4* and *STK3*) phosphorylate **LATS1/2**.
2. LATS keeps two related proteins, **YAP** and **TAZ** (gene *WWTR1*), out of the nucleus.
3. When YAP/TAZ *do* enter the nucleus, they work with **TEAD** proteins to turn on growth programs.

“Phosphorylate” just means “attach a phosphate tag.” That tag often acts as an on/off switch.

YAP and TAZ are famously sensitive to **cytoskeletal tension**. When the actin mesh is taut, YAP tends to go nuclear and drive growth. When tension drops, YAP stays out. So Hippo and actin sit in the same mechanical circuit — they share the job of coupling “how the cell feels” to “whether the cell grows.”

That circuit diagram does **not** tell you which proteins physically meet.

### Why yeast is in this project

Baker’s yeast (*Saccharomyces cerevisiae*) is a single-celled fungus that is easy to grow and genetically manipulate. It has **no YAP or TAZ**. It does have a related signaling module called **RAM/MOR**, organized around the kinases **Cbk1** and **Kic1**.

RAM/MOR does analogous work at the **actin-rich bud cortex** — the growing tip of a yeast cell that is about to divide. The analogy is useful as a wiring diagram: both systems couple polarity and cytoskeleton to growth decisions. It is not a claim that yeast Cbk1 “is” human YAP.

Including yeast lets the atlas ask: if the *circuit* is conserved, which *protein pairs* look similar, and which catalogs actually document them?

Human and yeast proteins are kept in two separate subgraphs. They never form a cross-species edge in this analysis. The full network is simply the two species graphs placed side by side.

---

## Key biological terms

- **Protein–protein interaction (PPI).** Two proteins that associate — they bind directly, sit in the same complex, or genetically depend on each other. “Interact” in databases is broader than “touch.”
- **Physical interaction.** Evidence that the proteins are in the same complex or close in space (for example affinity purification or proximity labeling). Still not always a mapped contact surface.
- **Genetic interaction.** Changing one gene alters the effect of changing another. Useful in yeast; it does not mean the proteins bind.
- **Kinase.** A protein that attaches phosphate tags to other proteins.
- **GTPase** (here: RHOA, CDC42, RAC1, and yeast Rho1/Cdc42). Molecular switches that help control actin and cell polarity. They show up a lot because they are **hubs** — proteins with many catalogued partners.
- **Nucleus / cytoplasm / cortex.** Compartments of the cell. Nucleus holds DNA; cytoplasm is the rest of the interior; cortex is the actin-rich shell under the membrane. YAP shuttles between cytoplasm and nucleus, which is why “nuclear vs cortical” is not treated as a hard conflict in this atlas.
- **Seed protein.** A protein we start from on purpose (the Hippo/RAM list and the actin list). Partners of seeds are downloaded to give the neighborhood **context**, but they are not extra atlas rows.
- **Ortholog.** A protein in another species that looks like it came from the same ancestral gene. Yeast has no YAP/TAZ ortholog; RAM/MOR is treated as a *functional analog*, not a one-to-one match.

---

## What data sources were used — and why they disagree

Imagine three phone books for the same city, each compiled with different rules. That is STRING, BioGRID, and IntAct.

### STRING

[STRING](https://string-db.org) is a **functional association** network. An edge (a line between two proteins) means “these two are associated in some sense,” not necessarily “they bind.”

STRING’s **combined score** (0 to 1, often written as 0–1000) mixes several **channels**:

- experiments and curated databases
- co-expression (genes that turn on together)
- co-occurrence across species
- **text mining** (proteins mentioned near each other in papers)

A commonly used cutoff is **combined score ≥ 700**. A pair can sit above 700 because of text mining and co-annotation even with **no** curated physical assay. The reverse also happens: a pair can be in BioGRID and still score below 700 in STRING.

This project uses STRING v12 for human (taxonomy ID 9606) and yeast (4932). High-confidence partners (≥ 700, at most 20 per seed) define a **359-protein neighborhood** with **2,484** undirected edges. Sequences and locations come from [UniProt](https://www.uniprot.org), the standard protein catalog.

### BioGRID

[BioGRID](https://thebiogrid.org) is a **curated** database. Curators read papers and record interactions with:

- whether the evidence is **physical** or **genetic**
- the **assay name** (for example Affinity Capture–MS)
- PubMed IDs

This run used BioGRID release 5.0.260 organism files for human and *S. cerevisiae*.

### IntAct

[IntAct](https://www.ebi.ac.uk/intact) is another curated molecular-interaction database (from EMBL-EBI). Records include detection method, interaction type, and publications. This project queried IntAct through PSIQUIC and joined on UniProt accession.

### Why catalogs disagree

They are not trying to encode the same thing.

| Resource | What an “edge” usually means |
|---|---|
| STRING | Functional association; score mixes experiments, databases, and text mining |
| BioGRID | A curated physical or genetic record with an assay and a paper |
| IntAct | A curated molecular interaction with method and paper |

A pair can be a high-confidence STRING edge without a physical assay. It can sit in BioGRID and miss STRING’s 700 cutoff. Yeast RAM × actin evidence, when it exists at all, is often **genetic**. Human Hippo × actin evidence is mostly **co-complex or proximity** experiments, not reconstituted two-protein binding.

### What an “evidence atlas” is

An **evidence atlas** is a table that does not pick a winner among those catalogs. For every pair of interest it records:

- what STRING says (including channel scores below 700)
- whether BioGRID has a physical or genetic record
- whether IntAct has a record
- a coarse check of whether the two proteins could reasonably share space in the cell
- a computer rank (GraphSAGE), kept as an extra column

Each pair then gets **one exclusive class** so you can sort the table without mixing apples and oranges.

### The four evidence classes

Every one of the 160 Hippo × actin pairs is assigned exactly one of these:

1. **`physical_curated`** — BioGRID physical **or** IntAct record exists, even if STRING’s combined score is below 700. This is “someone curated a physical-style experiment,” not “we reconstituted the pair in a test tube.” Yeast genetic-only pairs are **not** promoted into this class.
2. **`string_functional_only`** — STRING combined score ≥ 700, and no curated physical record. In *this* neighborhood, no Hippo × actin pair landed here: the few STRING ≥ 700 pairs were either physical or flagged as artifact-risk.
3. **`unreported`** — absent from BioGRID physical, IntAct, **and** STRING ≥ 700. The pair may still have a low STRING score or a genetic-only yeast record.
4. **`artifact_risk`** — not curated physical, and at least one red flag: a localization conflict, STRING support that is **text-mining-only** at ≥ 700, or both proteins so well-connected that their **degree product** is at or above the atlas 90th percentile. The cutoff is conservative on purpose. It will sometimes flag real biology that happens to involve a hub.

**Genetic-only** yeast pairs (BioGRID genetic, no physical BioGRID, no IntAct) get a separate flag. They stay `unreported` rather than `physical_curated`.

---

## How the computational part works

You can skip the formulas. The idea is simple.

### Step 1 — Build a neighborhood, not the whole proteome

Starting from 10 human Hippo proteins and 10 human actin proteins, plus 6 yeast RAM proteins and 10 yeast actin proteins (36 seeds), the pipeline downloads STRING partners at combined score ≥ 700 (max 20 per seed). That yields **359 proteins**. Tight around the seeds; not a dump of every human protein.

The **atlas itself** is only the same-species Hippo × actin combinations among the 36 seeds:

- 10 × 10 = **100 human pairs**
- 6 × 10 = **60 yeast pairs**
- **160 pairs** total

STRING partners stay in the graph as context. They are not extra rows in the atlas.

### Step 2 — Give each protein a “fingerprint” from its sequence (ESM-2)

A **protein language model** is a neural network trained on millions of protein sequences, the way a text model is trained on sentences. It learns which amino-acid patterns tend to co-occur.

This project uses a frozen **ESM-2** model (`facebook/esm2_t12_35M_UR50D`): 12 layers, 35 million parameters. “Frozen” means it was **not** retrained on interaction labels. Each protein becomes one 480-number vector by averaging the model’s residue-level outputs. Sequences longer than 1,022 amino acids were truncated (89 proteins).

Think of the vector as a fingerprint of sequence family and likely biochemical role — not a 3D structure and not a binding prediction by itself.

### Step 3 — Learn from the network with GraphSAGE

A **graph** here is dots (proteins) connected by lines (STRING associations). A **graph neural network (GNN)** updates each protein’s fingerprint by looking at its neighbors, then its neighbors’ neighbors.

**GraphSAGE** is one such GNN. In plain language:

> To describe protein A, mix A’s own sequence fingerprint with the average fingerprint of the proteins STRING already links to A.

Two layers of that mixing produce a 64-number description per protein. A small extra network (an **MLP**, multilayer perceptron) then looks at a pair of descriptions and outputs a probability: “does this look like a STRING edge?”

Training uses known STRING ≥ 700 edges, split 80/10/10 within each species. The model only “sees” training edges when it talks to neighbors, so the test pairs are a fair quiz. Dummy non-edges are random same-species pairs that are not in STRING.

That quiz answers: *can we recover held-out STRING associations?* It does **not** answer: *do these proteins bind?*

### Step 4 — Rank Hippo × actin pairs that STRING does not already list

After training, every Hippo × actin pair gets a GraphSAGE probability. The interesting ranks are the **156 pairs absent from STRING at ≥ 700**. Those ranks are written into the atlas.

A second evaluation (**physical-label benchmark**) retrains against BioGRID/IntAct physical edges among the 359 proteins, still using STRING only as the context graph. That checks whether topology or sequence recover *curated physical* labels — a stricter question.

---

## Key machine-learning and network terms

- **Graph / network.** Proteins as nodes; associations as edges.
- **Degree.** How many partners a protein has in this STRING neighborhood. Actin and Rho-family GTPases tend to have high degree.
- **Degree product.** Degree of protein A times degree of protein B. Two hubs multiplied together look “important” to many algorithms even if they never meet.
- **Shared neighbors (common neighbors).** Proteins linked to both A and B. If A and B have many friends in common, simple rules already guess they might be associated. That is **neighborhood overlap**.
- **Link prediction.** Guessing missing edges. Here: “would STRING have drawn this line?” or, in the benchmark, “does a curated physical record exist?”
- **Jaccard / Adamic–Adar / preferential attachment.** Classic scoring rules that use only the graph (shared neighbors or hub-ness). No sequence, no learned weights. **Adamic–Adar** down-weights very popular shared neighbors.
- **AUROC.** A 0–1 score for ranking. 0.5 is coin-flip on a balanced quiz; 1.0 is perfect. Useful for comparing methods; not a percent of biology discovered.
- **Negative example.** A pair labeled “not an edge” for training. Here they are **random non-edges**, not experimentally confirmed non-interactions. That usually makes AUROC look a bit optimistic.
- **Hub bias.** Models trained on association networks often give high scores to popular proteins. GraphSAGE ranks in this atlas **correlate** with degree product (Spearman 0.58) and shared-neighbor count (0.46). Read high ranks with that in mind.

**Spearman correlation** is a 0-to-1 (or −1-to-1) measure of whether two rankings move together. 0.58 means “higher GraphSAGE score tends to go with more hub-like pairs,” not “the model is 58% correct.”

---

## What the main results actually mean

### The catalogs do not tell one story

Of **160** Hippo × actin pairs:

- **4** already appear in STRING at combined score ≥ 700.
- Of the other **156**:
  - **13 (8.3%)** already have BioGRID physical or IntAct records — they were “missing” from STRING ≥ 700, not missing from biology.
  - **11 (7.1%)** are flagged `artifact_risk` (mostly human hubs; STRING support, when present, is often text mining).
  - **132 (84.6%)** are `unreported` at these cutoffs.

Yeast is thinner: one curated physical pair among STRING-absent yeast pairs (Act1–Tao3), ten genetic-only pairs, and the rest unreported. Human STRING-absent pairs include 12 curated physical records.

Across the whole 359-protein set, BioGRID and IntAct together contribute 3,124 physical edges; 1,863 of those are **not** STRING ≥ 700, and 1,223 STRING ≥ 700 edges have **no** physical record in this join. Overlap exists. The resources are not interchangeable.

### GraphSAGE mostly recovers neighborhood overlap on STRING

On held-out STRING edges, GraphSAGE reached AUROC **0.910**. Adamic–Adar, which only uses shared neighbors, reached **0.904**. On a small, dense graph, “who shares friends with whom” already recovers most STRING edges. Sequence features add a little, not a leap.

### Top candidates — read the class column, not only the rank

**Act1–Kic1** (yeast) is rank 1 among STRING-absent pairs (probability 0.947). Class: **`unreported`**. No BioGRID physical record, no IntAct record, STRING combined score 0.176 (text mining). Kic1 works at the actin-rich bud cortex, so a real association with Act1 would place RAM/MOR on the cortical cytoskeleton in a compact way. That is why the paper lingers on it. It is a reasonable wet-lab target. It is **not** a demonstrated mechanism. A second model trained on physical labels is unimpressed (median rank 29.5; never in the top 10).

**YAP1–ACTB** (human) is rank 2 (0.920). Class: **`physical_curated`**. BioGRID already has Affinity Capture–MS from two papers. STRING combined score is 0.515 — below 700 — so a STRING-only pipeline would have called it “novel.” The atlas’s job is to stop that mistake. A co-complex with β-actin would fit YAP’s known sensitivity to cortical tension. It is still an inference from co-complex, not a mapped binding interface.

Lower on the list, some pairs are already proximity labeling or AP–MS (VCL–YAP1, STK4–CDC42); some are unreported (VCL–NF2, Cbk1–Rho1); some are hub/text-mining **artifact-risk** (WWTR1–ACTB, YAP1–CDC42).

### The physical-label benchmark (why the paper is cautious)

When the task is “recover BioGRID/IntAct physical edges,” a **logistic regression on degree and common neighbors** matches or beats GraphSAGE (AUROC about 0.74 vs 0.72). Sequence features help more when hub-ness is taken off the table (degree-matched negatives). Shared-neighbor tricks fail if the test proteins were never seen in training.

So: GraphSAGE is a useful ranking column. It is not better than simple topology at this scale when the label is “STRING edge” or even “curated physical edge” with random negatives.

### The takeaway of the paper

Inside this 359-protein neighborhood there are 160 Hippo × actin pairs. Most of the 156 that miss STRING ≥ 700 are truly unreported at these cutoffs; a minority already have curated physical records that STRING’s cutoff hid. Learned ranks track hub-ness. **The useful object is the evidence atlas.** The model scores are a column in that table, not the point of the work.

If a pair is worth chasing, yeast two-hybrid, affinity purification–mass spectrometry, or proximity labeling are still the right experiments. They are not a practical way to screen every combination by hand — and they are not what a STRING functional edge means.

---

## How to explore or reproduce the work

### What lives in this repository

| Place | What it is |
|---|---|
| `src/` | Python pipeline: download, embeddings, training, atlas, benchmark, figures |
| `data/raw/` | Downloaded STRING/UniProt graph and BioGRID/IntAct evidence files |
| `data/processed/` | Atlas, embeddings, model weights, metrics |
| `figures/` | Paper figures (t-SNE, ROC, subgraphs, evidence classes) |
| `tests/` | Unit tests (no live BioGRID/IntAct download in `make test`) |
| `manuscript.md` | The full paper |
| `README.md` | Short technical summary and command list |

You need **Docker**. There is no local virtualenv. On Mac, Docker runs **CPU** PyTorch (not the laptop GPU). The 359-node graph is small enough that this is fine.

### High-level commands

```bash
docker compose build
make test          # unit tests
make fetch         # STRING + UniProt download
make embed         # ESM-2 fingerprints for each protein
make train         # train GraphSAGE on STRING edges
make viz           # network figures
make atlas         # build the evidence atlas (may also run the physical-label benchmark)
make benchmark     # physical-label benchmark only (needs atlas outputs)
```

A one-shot reproduction path from a clean image is:

```bash
docker compose build
make fetch embed train viz atlas
```

To build the atlas CSV without the longer benchmark:

```bash
docker compose run --rm -e HIPPO_SKIP_BENCHMARK=1 --entrypoint python pipeline -m src.evidence_atlas
```

### The main output: `hippo_actin_atlas.csv`

Path: **`data/processed/hippo_actin_atlas.csv`**

One row per Hippo × actin pair (160 rows). Useful columns for a first pass:

| Column | Meaning |
|---|---|
| `species`, `protein_a`, `protein_b` | Which organism and which two proteins |
| `evidence_class` | `physical_curated`, `string_functional_only`, `unreported`, or `artifact_risk` |
| `graphsage_probability` / `graphsage_rank` | Model score and rank (higher probability = model thinks it looks like a STRING edge) |
| `string_combined` | STRING combined score (0–1). Below 0.7 is “absent” at the project cutoff |
| `dominant_channel` | Which STRING channel contributes most (often `textmining`) |
| `biogrid_physical`, `biogrid_genetic`, `intact` | Whether those catalogs have a record |
| `assays` | BioGRID assay names when present |
| `pubmeds` | Paper IDs supporting catalog records |
| `degree_product`, `shared_neighbors` | Hub-ness and neighborhood overlap |
| `localization_overlap` | `compatible`, `unclear`, or conflicting compartments |
| `genetic_only` | Yeast-style genetic evidence without a physical record |

Sort by `graphsage_rank_among_absent` to see the STRING-absent hit list. Always read `evidence_class` next to the rank.

Related files:

- `data/processed/top_predicted_interactions.csv` — top 50 STRING-absent pairs
- `data/processed/atlas_stats.json` — counts and class breakdowns
- `figures/figure4_evidence_classes.png` — visual of the four classes

---

## Glossary (short)

| Term | Plain meaning |
|---|---|
| Actin / cytoskeleton | Protein filaments that give the cell shape and sense mechanical force |
| Hippo pathway | A growth-control pathway; YAP/TAZ activity tracks cytoskeletal tension |
| RAM/MOR | Yeast signaling module (Cbk1, Kic1, and partners) that is Hippo-like at the bud cortex |
| PPI | Protein–protein interaction; association, not always direct binding |
| STRING | Functional-association database; mixed evidence, combined score |
| BioGRID / IntAct | Curated interaction databases with assays and papers |
| Evidence atlas | One table joining those resources plus model ranks and quality flags |
| ESM-2 | Protein language model; turns a sequence into a numeric fingerprint |
| GraphSAGE | Neural net that updates each protein using its network neighbors |
| Link prediction | Ranking possible missing edges in a network |
| Degree / hub | Number of partners; hubs have many and can inflate scores |
| AUROC | Ranking quality versus a coin-flip baseline of 0.5 |

---

## Where to go next

1. **Skim the atlas.** Open `data/processed/hippo_actin_atlas.csv`. Filter to `unreported` if you want pairs with no catalog support at these cutoffs. Filter to `physical_curated` if you want pairs STRING ≥ 700 missed.
2. **Look at Figure 4.** `figures/figure4_evidence_classes.png` shows the class counts and how GraphSAGE scores track degree product.
3. **Read the paper** in `manuscript.md` (or `manuscript.pdf`) once the vocabulary here feels familiar. The Discussion is the honest part: what the ranks do and do not mean.
4. **Reproduce** with Docker (`make fetch embed train viz atlas`) if you want the numbers regenerated rather than trusted from disk.
5. **If you take a pair to the lab**, treat GraphSAGE as a triage aid. Confirm with an orthogonal experiment. The atlas will tell you whether you are testing a true unknown or rediscovering a record another database already had.

You do not need to become a Hippo biologist or a graph-network expert to use this repository. You need the atlas, the four class labels, and a healthy suspicion of any score that loves hubs.
