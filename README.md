# CSL7110 — Assignment 4: Clustering and PageRank

**Name:** Kakarla Sai Swaroop  
**Roll No:** M25DE1023  
**Email:** m25de1023@iitj.ac.in  
**Course:** CSL7110 — Big Data Analytics  


---

## Overview

This assignment implements three core big data algorithms:

| Part | Topic | Algorithm |
|------|-------|-----------|
| 1 | Clustering | Farthest-First Traversal (K-Center) + K-Means++ |
| 2 | Web Search | Inverted Index with custom Hash Table |
| 3 | Graph Analytics | PageRank on Apache Spark |

---

## Repository Structure

```
CSL7110-Assignment4/
│
├── M25DE1023_Kakarla_Sai_Swaroop_Assignment4_CSL7110.ipynb   # Main notebook (all 3 parts)
├── README.md                                                   # This file
│
├── Assignment 4- datasets/
│   ├── Q1- UCI Spam clustering/
│   │   └── spambase.data                  # 4601 x 58 email spam dataset
│   └── Q2- webSearch/
│       ├── actions.txt                    # Search queries
│       ├── answers.txt                    # Expected outputs
│       └── webpages/                      # 7 webpage text files
│           ├── stack_datastructure_wiki
│           ├── stack_cprogramming
│           ├── stack_oracle
│           ├── stackoverflow
│           ├── stacklighting
│           ├── stackmagazine
│           └── references
│
└── graph/                                 # PageRank graph files (auto-downloaded in notebook)
    ├── small.txt                          # 100 nodes, 950 edges (validation)
    └── whole.txt                          # 1000 nodes, 8161 edges (main experiment)
```

---

## Part 1 — Clustering

### Dataset
- **UCI Spambase Dataset** — 4,601 email records, 57 features (last column label dropped)
- Features represent word/character frequencies and capital-letter run statistics

### Algorithms Implemented

#### `readVectorsSeq(filename)`
Reads the CSV dataset and returns a list of NumPy arrays (one per point).

#### `kcenter(P, k)` — Farthest-First Traversal
- Greedily selects k centers by always picking the point farthest from existing centers
- **Time complexity:** O(|P| × k)
- **Guarantee:** 2-approximation to optimal k-center

#### `kmeansPP(P, k)` — K-Means++ Seeding
- Selects centers using D² weighted sampling — farther points are more likely chosen
- **Time complexity:** O(|P| × k)
- **Guarantee:** O(log k) approximation to k-means objective

#### `kmeansObj(P, C)` — K-Means Objective
- Computes average squared distance from each point to its nearest center
- Lower = better clustering quality

### Results

| Experiment | Description | Time | kmeansObj |
|---|---|---|---|
| Exp 1 | `kcenter(P, k=3)` | 0.0407s | 290,451.57 |
| Exp 2 | `kmeansPP(P, k=3)` | 0.0446s | 151,995.45 |
| Exp 3 | `kcenter(P, k1=10)` → `kmeansPP(X, k=3)` | 0.4821s | 264,462.90 |

**Key finding:** K-Means++ (Exp 2) gives the best clustering quality. The coreset approach (Exp 3) is a scalable alternative — increasing k1 improves quality.

---

## Part 2 — Web Search: Inverted Index

### Classes Implemented

| Class | Role |
|---|---|
| `Position` | Stores `<PageEntry, word_index>` tuple |
| `WordEntry` | All positions for a word across pages |
| `PageIndex` | Per-page map: word → WordEntry |
| `PageEntry` | Reads a file and builds its index |
| `MyHashTable` | Custom hash table (polynomial rolling hash, base 31) |
| `InvertedPageIndex` | Global aggregator across all pages |
| `SearchEngine` | Processes action commands |

### Text Processing Rules
- All words converted to **lowercase**
- **Stop words** skipped in index but counted in position numbering
- **Punctuation** `{ } [ ] < > = ( ) . , ; ' " ? # ! - :` replaced with space
- **Singular/plural** normalization: `stacks→stack`, `structures→structure`, `applications→application`
- **Query normalization:** non-alpha chars stripped before lookup (e.g. `C++` → `c`)

### Query Results

All **11 / 11** outputs match `answers.txt` exactly ✓

| Query | Output |
|---|---|
| `queryFindPagesWhichContainWord delhi` | No webpage contains word delhi |
| `queryFindPagesWhichContainWord stack` | stack_datastructure_wiki |
| `queryFindPagesWhichContainWord wikipedia` | stack_datastructure_wiki |
| `queryFindPositionsOfWordInAPage magazines stack_datastructure_wiki` | Webpage ... does not contain word magazines |
| `queryFindPagesWhichContainWord allain` (after adding stack_cprogramming) | stack_cprogramming |
| `queryFindPagesWhichContainWord C` | stack_cprogramming |
| `queryFindPagesWhichContainWord C++` | stack_cprogramming |
| `queryFindPagesWhichContainWord jdk` | stack_oracle |
| `queryFindPagesWhichContainWord function` | stack_cprogramming, stack_datastructure_wiki, stackoverflow |
| `queryFindPagesWhichContainWord magazines` | stackmagazine |

---

## Part 3 — PageRank on Spark

### Algorithm
Iterative power iteration for 40 steps with β = 0.8:

```
r^(i) = ((1 - β) / n) * 1  +  β * M * r^(i-1)
```

Where M[j][i] = 1/deg(i) if edge (i→j) exists, else 0.

### Spark Implementation
```python
links = edges.groupByKey().mapValues(list).cache()

for i in range(40):
    contribs = links.join(ranks).flatMap(
        lambda kv: [(dst, β * kv[1][1] / len(kv[1][0])) for dst in kv[1][0]]
    )
    ranks = contribs.reduceByKey(lambda a, b: a + b) \
                    .mapValues(lambda v: (1 - β) / n + v)
```

### Results

**Small Graph Validation (`small.txt` — 100 nodes, 950 edges)**

| Rank | Node | Score |
|---|---|---|
| #1 | 53 | 0.035731 |
| #2 | 14 | 0.034171 |
| #3 | 40 | 0.033630 |
| #4 | 1 | 0.030006 |
| #5 | 27 | 0.029720 |

Top score: **0.0357** — matches expected ≈ 0.036 ✓

**Full Graph (`whole.txt` — 1000 nodes, 8161 edges)**  
Computed in **1.86 seconds** (PySpark, 40 iterations, β = 0.8)

| Rank | Node | Score |
|---|---|---|
| #1 | 263 | 0.002020 |
| #2 | 537 | 0.001943 |
| #3 | 965 | 0.001925 |
| #4 | 243 | 0.001853 |
| #5 | 285 | 0.001827 |

| Rank | Node | Score |
|---|---|---|
| #996 | 408 | 0.000388 |
| #997 | 424 | 0.000355 |
| #998 | 62 | 0.000353 |
| #999 | 93 | 0.000351 |
| #1000 | 558 | 0.000329 |

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Colab
2. Run the **Setup Cell** at the top — it mounts Google Drive, extracts the dataset, downloads graph files, and installs PySpark automatically
3. Run all cells in order (`Runtime → Run all`)

### Option 2 — Local Jupyter

```bash
# Install dependencies
pip install numpy pyspark jupyter

# Launch notebook
jupyter notebook M25DE1023_Kakarla_Sai_Swaroop_Assignment4_CSL7110.ipynb
```

Update the dataset paths in the notebook to point to your local directories.

### Dataset Sources
- **Spambase:** Included in the assignment zip (`spambase.data`)
- **Webpages:** Included in the assignment zip (`Q2- webSearch/webpages/`)
- **PageRank graphs:** Auto-downloaded in notebook from [pnijhara/PySpark-PageRank](https://github.com/pnijhara/PySpark-PageRank/tree/main/graph)

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Vector operations, clustering |
| `pyspark` | ≥ 3.0 | Distributed PageRank |
| `re` | stdlib | Query word normalization |
| `collections` | stdlib | defaultdict for adjacency lists |
| `time` | stdlib | Performance timing |

---

## Assumptions

1. The last column of `spambase.data` is always the class label and is dropped
2. The first center in `kcenter` is always `P[0]` (deterministic start)
3. Stop words, punctuation, and singular/plural lists from the assignment are treated as exhaustive
4. Non-alphabetic tokens (e.g. `C++`, numbers) are not stored in the index, but their positions are counted
5. Multiple directed edges between the same node pair are treated as a single edge in PageRank
6. Self-loops are removed from the PageRank graph before processing

---

## References

- Farthest-First Traversal: http://www.wikiwand.com/en/Farthest-first_traversal
- K-Means++ Paper: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- K-Means++ Slides: http://theory.stanford.edu/~sergei/slides/BATS-Means.pdf
- PageRank Dataset: https://github.com/pnijhara/PySpark-PageRank/tree/main/graph
