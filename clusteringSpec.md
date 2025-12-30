Great — this is a good moment to **lock the next layer cleanly**.
Below is a **handoff-ready spec sheet** for adding **K-means clustering (with silhouette-based k selection)** into your **existing embedding + dedup pipeline**, so that a *single run* produces:

> **deduplicated embeddings + cluster labels (+ centroids)**

This spec is written to fit **exactly** into the code you shared  and assumes **Google Colab (free tier)**.

---

# 📄 SPEC SHEET

## K-Means Clustering Layer (Post-Dedup, Pre-Save)

---

## 0. Scope & Goal

Add a **semantic clustering stage** to the current pipeline such that:

* Embeddings are generated (GPU)
* Near-duplicate comments are removed (cosine dedup, CPU)
* **K-means clustering is run on deduplicated embeddings**
* Optimal number of clusters is selected using **silhouette score**
* Final output file contains:

  * embeddings
  * comment IDs
  * cluster labels
  * cluster centroids

This clustering must run **before saving** the `.npz` file.

---

## 1. Execution Environment (MANDATORY)

* Google Colab (free tier)
* CPU for clustering (GPU not required)
* sklearn available
* Dataset size: ~1k–50k comments after dedup

⚠️ No GPU clustering, no FAISS, no heavy PCA in first version.

---

## 2. Placement in Existing Pipeline (STRICT)

### Current flow in `process_file()`

```
comments
→ embedding (GPU, batched)
→ concatenate embeddings
→ cosine_deduplicate()
→ save npz
```

### Required new flow

```
comments
→ embedding
→ concatenate
→ cosine_deduplicate()
→ 🔹 k-means clustering + silhouette search
→ save npz (with cluster data)
```

Clustering must run **after deduplication** and **before saving**.

---

## 3. Inputs to Clustering

From deduplication output:

* `dedup_embeddings`: NumPy array `(M, D)`
* `dedup_comment_ids`: list `(M,)`
* `dedup_comments`: list `(M,)`

Where:

* embeddings are **L2-normalized**
* order is preserved

---

## 4. Clustering Algorithm Choice

### Algorithm

* `MiniBatchKMeans` (from `sklearn.cluster`)

### Rationale

* Faster and more memory-efficient than full KMeans
* Sufficient quality for text embeddings
* Stable in Colab

---

## 5. Valid Range for Number of Clusters (k)

### Important constraint

❌ **k = 1 is NOT allowed**
Silhouette score is undefined for 1 cluster.

### Allowed range

```
k ∈ [2, K_max]
```

### K_max heuristic (MANDATORY)

```python
K_max = min(20, int(np.sqrt(M)))
```

Where `M = number of deduplicated embeddings`.

This avoids:

* over-fragmentation
* slow silhouette computation
* meaningless micro-clusters

---

## 6. Silhouette-Based Model Selection

### For each k in range:

1. Fit MiniBatchKMeans
2. Predict cluster labels
3. Compute `silhouette_score(X, labels, metric="euclidean")`

### Store:

* k
* silhouette score

### Selection rule

* Choose k with **maximum silhouette score**
* If scores plateau (difference < ~0.01), prefer **smaller k**

This keeps clusters interpretable for creators.

---

## 7. Final Clustering Run

Once `best_k` is selected:

* Re-fit `MiniBatchKMeans(n_clusters=best_k)`
* Generate:

  * `cluster_labels`: `(M,)`
  * `cluster_centers_`: `(best_k, D)`

These represent:

* cluster assignment per comment
* semantic centroid per theme

---

## 8. Outputs to Save (EXTENDED NPZ FORMAT)

The final `.npz` file must contain:

```python
np.savez_compressed(
    output_path,
    embeddings=dedup_embeddings,      # (M, D)
    ids=dedup_comment_ids,            # (M,)
    labels=cluster_labels,            # (M,)
    centroids=cluster_centers,         # (K, D)
)
```

This enables:

* cluster-level analysis
* centroid-nearest sampling
* LlamaIndex integration without recomputation

---

## 9. Logging Requirements (IMPORTANT)

Add logs at each stage:

* Before search:

```
🔍 Searching for optimal number of clusters (k=2…K_max)
```

* Per k:

```
📊 k=6 → silhouette=0.421
```

* After selection:

```
✅ Optimal clusters selected: k=6 (silhouette=0.421)
```

This is essential for:

* debugging
* demos
* trust in the system

---

## 10. Performance Constraints (Colab-Safe)

* Use `MiniBatchKMeans`
* Avoid PCA in v1
* Avoid silhouette on very large k
* No GPU usage required
* Expected runtime:

  * ~seconds for 1k–10k points
  * ~1–2 minutes for ~30k points

---

## 11. Non-Goals (Explicit)

❌ Emotion analysis
❌ LLM usage
❌ Cluster labeling
❌ Temporal modeling
❌ Hierarchical clustering

This spec is **clustering only**.

---

## 12. Acceptance Criteria

This feature is complete when:

* A single run produces embeddings + clusters
* `.npz` contains labels and centroids
* k is selected automatically (no hard-coding)
* Works end-to-end in Colab
* No re-loading of saved embeddings required

---

## 13. Design Principle (For the Coding Agent)

> **Dedup removes noise.
> K-means finds structure.
> LLMs explain meaning.**

Do not mix these responsibilities.

---

## 14. Why This Is the Right Next Step

This clustering layer creates **semantic themes**, which become the *unit of insight* for:

* emotion aggregation
* creator feedback
* LlamaIndex summarization
* “what to fix / ignore / double-down on”

You are now building a **meaning pipeline**, not just analytics.

---

