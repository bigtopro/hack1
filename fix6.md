Perfect — this is exactly the moment for a **clean, authoritative modification spec**.
Below is a **handoff-ready spec sheet** your coding agent can follow **line-by-line**, based on your **current code**  and the decisions we’ve locked in.

This spec removes the pathological slowdowns, aligns with **Method 2 (√N)**, and preserves everything that already works.

---

# 📄 MODIFICATION SPEC SHEET

## Fast & Scalable Clustering for 60k+ Comments

### Context

* Dataset size: ~60,000 comments
* Embeddings already L2-normalized (E5 model)
* Goal: semantic indexing → emotion-conditioned LLM analysis
* Constraints: must run fast on Colab CPU/GPU

---

## 🔒 FINAL DESIGN DECISIONS (DO NOT DEVIATE)

1. **Clustering k selection**:
   → **Method 2: √N heuristic, capped**

2. **Deduplication**:
   → **REMOVE ENTIRELY**

3. **Silhouette score / k search**:
   → **REMOVE ENTIRELY**

4. **Clustering algorithm**:
   → `MiniBatchKMeans` with fixed k

Clustering is an **indexing step**, not a mathematical optimization task.

---

## 1️⃣ REMOVE DEDUPLICATION COMPLETELY (CRITICAL)

### ❌ DELETE this function entirely

```python
def cosine_deduplicate(...)
```

### ❌ REMOVE this call in `process_file()`:

```python
dedup_embeddings, dedup_comment_ids, dedup_comments = cosine_deduplicate(...)
```

### ❌ REMOVE all references to:

* `dedup_embeddings`
* `dedup_comment_ids`
* `dedup_comments`
* deduplication statistics/logs

---

### ✅ REPLACE with direct usage of full embeddings

Immediately after:

```python
embeddings_array = np.vstack(all_embeddings)
```

Add:

```python
final_embeddings = embeddings_array
final_ids = comment_ids
final_comments = comments
```

These variables are now the **single source of truth**.

---

## 2️⃣ REMOVE SILHOUETTE-BASED CLUSTERING (MANDATORY)

### ❌ DELETE this entire function:

```python
def kmeans_clustering_with_silhouette(...)
```

### ❌ REMOVE imports:

```python
from sklearn.metrics import silhouette_score
```

### ❌ REMOVE all logs referencing:

* silhouette
* “optimal k”
* silhouette scores
* k search loops

---

## 3️⃣ IMPLEMENT METHOD 2 (√N) CLUSTERING

### ✅ ADD this helper function

```python
def kmeans_clustering_fixed_k(
    embeddings: np.ndarray,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fast clustering using sqrt(N) heuristic for k.
    """
    n_samples = len(embeddings)

    if n_samples == 0:
        return np.array([]), np.array([]), 0

    if n_samples == 1:
        return np.array([0]), embeddings[:1], 1

    # Method 2: sqrt(N) capped
    k = min(50, max(2, int(np.sqrt(n_samples))))

    logger.info(f"🎯 Clustering with fixed k={k} (sqrt heuristic)")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=2048,
        n_init=3
    )

    labels = kmeans.fit_predict(embeddings)

    return labels, kmeans.cluster_centers_, k
```

---

## 4️⃣ UPDATE `process_file()` CLUSTERING BLOCK

### ❌ REMOVE this entire section:

```python
logger.info("🔍 Starting clustering for deduplicated embeddings...")
cluster_labels, cluster_centers, best_k, best_silhouette = ...
```

### ✅ REPLACE with:

```python
logger.info(f"🔍 Starting clustering for {len(final_embeddings)} embeddings...")
cluster_labels, cluster_centers, k = kmeans_clustering_fixed_k(
    final_embeddings
)
logger.info(f"✅ Clustering completed with k={k}")
```

---

## 5️⃣ DISTANCE-TO-CENTROID (KEEP, BUT FIX INPUTS)

### ✅ UPDATE this loop to use `final_embeddings`

```python
distances_to_centroid = np.zeros(len(final_embeddings))
for i in range(len(final_embeddings)):
    centroid = cluster_centers[cluster_labels[i]]
    distances_to_centroid[i] = np.linalg.norm(final_embeddings[i] - centroid)
```

---

## 6️⃣ UPDATE NPZ SAVE FORMAT

### ❌ REMOVE dedup-related fields

### ✅ FINAL `.npz` save block should be:

```python
np.savez_compressed(
    output_path,
    embeddings=final_embeddings,
    ids=final_ids,
    labels=cluster_labels,
    centroids=cluster_centers,
    distances=distances_to_centroid
)
```

---

## 7️⃣ UPDATE CLUSTERING SUMMARY JSON

### ❌ REMOVE fields:

* `deduplicated_comments`
* `silhouette_score`
* `clustering_algorithm` mentioning silhouette

### ✅ REPLACE with:

```python
clustering_summary = {
    "input_file": str(input_path.name),
    "output_file": str(output_path.name),
    "total_comments": len(final_embeddings),
    "embedding_dimension": final_embeddings.shape[1],
    "num_clusters": int(k),
    "cluster_distribution": {
        str(cluster_id): int((cluster_labels == cluster_id).sum())
        for cluster_id in np.unique(cluster_labels)
    },
    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "clustering_algorithm": "MiniBatchKMeans (sqrt(N) heuristic)"
}
```

---

## 8️⃣ UPDATE FINAL LOG OUTPUT

### ❌ REMOVE all logs referencing:

* deduplication %
* silhouette scores
* “optimal k search”

### ✅ KEEP logs focused on:

* total comments
* number of clusters
* embedding dimension
* output file size

---

## 9️⃣ EXPECTED PERFORMANCE AFTER CHANGES

| Stage          | Before         | After             |
| -------------- | -------------- | ----------------- |
| Deduplication  | ~1 hour        | ❌ removed         |
| k search       | ~30 minutes    | ❌ removed         |
| Clustering     | minutes        | **seconds**       |
| Total pipeline | **>1.5 hours** | **~5–10 minutes** |

---

## 10️⃣ DESIGN RATIONALE (FOR THE AGENT)

* Deduplication destroys frequency signal → removed
* Silhouette is O(N²) → removed
* √N heuristic is industry-standard for embedding indexing
* Emotional + LLM analysis provides semantic precision later
* Clustering only needs to be *good enough*

---

## ✅ ACCEPTANCE CHECKLIST

The modification is correct when:

* No deduplication code exists
* No silhouette imports or logs exist
* k is computed as `min(50, sqrt(N))`
* 60k comments cluster in under 1 minute
* Output `.npz` joins cleanly with sentiment JSON
* Cluster sizes are non-trivial and interpretable

---

## Final note (important)

This change **does not reduce insight quality** —
it **dramatically increases throughput and stability**.
