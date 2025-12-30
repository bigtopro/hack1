Here’s a **clean, copy-paste–ready prompt** you can give directly to your coding agent.
It is written as an **implementation instruction**, not a discussion, and is aligned exactly with your current codebase and pipeline.

---

## 🔧 CODING AGENT INSTRUCTION PROMPT

### Feature: Distance-to-Centroid Computation & Storage (Post-KMeans)

### Context

We already have a working pipeline that:

1. Embeds YouTube comments
2. Deduplicates embeddings using cosine similarity
3. Runs K-means clustering with silhouette-based optimal `k`
4. Saves embeddings, cluster labels, and centroids into a `.npz` file

We now want to **add distance-to-centroid computation** so that each comment can be ranked as:

* representative (near centroid)
* extreme / outlier (far from centroid)

This data will be used later by LlamaIndex for cluster analysis.

---

### Objective

Compute the **distance of each deduplicated comment embedding to its assigned cluster centroid**, and save this distance in the output file.

This must be done **after final K-means clustering** and **before saving the `.npz` file**.

---

### Technical Requirements

#### 1. Distance Definition

* Use **Euclidean distance** between:

  * comment embedding `x`
  * its cluster centroid `c`
* Embeddings are already L2-normalized, so Euclidean distance is acceptable and consistent.

Formula:

```python
distance = np.linalg.norm(x - c)
```

---

#### 2. Where to Add This Logic

Inside `process_file()`:

After this step:

```python
cluster_labels, cluster_centers, best_k, best_silhouette = kmeans_clustering_with_silhouette(
    dedup_embeddings
)
```

Add logic to compute distances **for each embedding** using its assigned cluster label.

---

#### 3. Required Output Array

Create a NumPy array:

```python
distances_to_centroid  # shape: (M,)
```

Where:

* `M = number of deduplicated embeddings`
* `distances_to_centroid[i]` corresponds to:

  * `dedup_embeddings[i]`
  * `cluster_labels[i]`

---

#### 4. Efficient Implementation (Vectorized Preferred)

Expected logic:

```python
distances = np.zeros(len(dedup_embeddings))

for i in range(len(dedup_embeddings)):
    centroid = cluster_centers[cluster_labels[i]]
    distances[i] = np.linalg.norm(dedup_embeddings[i] - centroid)
```

Vectorized or partially vectorized implementations are acceptable, but **clarity > micro-optimization**.

---

#### 5. Save Distances in Output File

Modify the final save step to include distances:

```python
np.savez_compressed(
    output_path,
    embeddings=dedup_embeddings,
    ids=dedup_comment_ids,
    labels=cluster_labels,
    centroids=cluster_centers,
    distances=distances_to_centroid
)
```

This ensures downstream consumers can:

* rank comments within clusters
* extract representative vs extreme cases
* pass structured inputs to LlamaIndex

---

#### 6. Logging (Required)

Add one log line after computing distances:

```text
📐 Computed distance-to-centroid for all clustered comments
```

No per-comment logging.

---

### Non-Goals (Do NOT implement)

* ❌ No sentiment analysis
* ❌ No LLM usage
* ❌ No re-clustering
* ❌ No distance normalization or scaling
* ❌ No removal of outliers at this stage

This task is **compute + store only**.

---

### Acceptance Criteria

This task is complete when:

* Output `.npz` file contains a `distances` array
* `len(distances) == len(embeddings) == len(labels)`
* Pipeline runs end-to-end in Google Colab
* No additional memory spikes or GPU usage introduced

---

### Design Principle (Do Not Violate)

> **Clusters define themes.
> Distance defines representativeness.
> Do not mix these responsibilities.**

---
