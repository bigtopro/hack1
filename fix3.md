
---

# 📄 SPEC SHEET

## Unified Embedding → Dedup → Clustering Pipeline (Single-Transaction)

### Version

**Pipeline v2 – Unified Semantic Finalization**

---

## 1. Objective

Refactor the existing monitoring + embedding pipeline so that **each input `comments.json` file is processed as a single semantic transaction**, producing **one final `.npz` file** that contains:

* embeddings
* deduplicated comment IDs
* cluster labels
* cluster centroids
* distance-to-centroid

Clustering must **always** run after deduplication, using the **same temporary state**, and must **never be skipped**.

---

## 2. Core Design Principle (MANDATORY)

> **A file is not “processed” until all semantic stages are complete.**

This includes:

* embedding
* deduplication
* clustering
* distance computation

Partial outputs (e.g. embeddings only) must never be treated as final.

---

## 3. High-Level Pipeline Flow (NEW)

For each `*_comments.json` file:

```
1. Detect new JSON file
2. Create temp directory: .tmp_<file_stem>/
3. Embed comments in batches → save batch_*.npz in temp dir
4. Reconstruct full embedding matrix from temp dir
5. Deduplicate embeddings (cosine-based)
6. Run K-means clustering on deduplicated embeddings
7. Compute distance-to-centroid for each comment
8. Save ONE final .npz file
9. Delete temp directory
10. Mark file as processed
```

---

## 4. Temp Directory Semantics (CRITICAL)

### Temp directory: `.tmp_<file_stem>/`

This directory represents **work in progress**.

Rules:

* Used ONLY for **embedding batch checkpoints**
* May persist across restarts for resume
* MUST be deleted after successful final save
* MUST NOT contain semantic outputs (clusters, distances, etc.)

If `.tmp_*` exists → resume embedding
If `.tmp_*` does not exist → fresh embedding run

---

## 5. Final Output Semantics (CRITICAL)

### Final file: `<file_stem>_comments_embeddings.npz`

This file represents **completed semantic processing**.

It MUST contain:

```python
embeddings  # shape (M, D)
ids         # shape (M,)
labels      # shape (M,)
centroids   # shape (K, D)
distances   # shape (M,)
```

Rules:

* Written **exactly once per run**
* Written **only after clustering and distance computation**
* Presence of this file means “pipeline completed successfully”

---

## 6. Changes Required in `process_new_files()`

### ❌ REMOVE this logic entirely

```python
if output_path.exists():
    logger.warning("⚠️ Embedding file already exists")
    continue
```

This logic is invalid in an evolving pipeline and causes clustering to be skipped.

---

### ✅ REPLACE with

```python
if output_path.exists():
    logger.warning("⚠️ Output exists — deleting and reprocessing")
    output_path.unlink()
```

This guarantees:

* clustering always runs
* schema upgrades are applied
* no silent skips occur

---

## 7. Changes Required in `process_file()`

### Required Structure (MANDATORY ORDER)

```text
process_file(input_path):

1. Initialize temp dir (.tmp_<file>)
2. Load comments
3. Resume or run embedding batches → batch_*.npz
4. Reconstruct embeddings from temp dir
5. Deduplicate embeddings
6. Run K-means clustering
7. Compute distance-to-centroid
8. Save final .npz
9. Delete temp dir
```

---

## 8. Deduplication Stage (UNCHANGED, BUT POSITION-LOCKED)

* Runs immediately after embeddings are reconstructed
* Operates fully in-memory
* Outputs:

  * dedup_embeddings
  * dedup_ids
  * dedup_comments

Deduplication must **not** save intermediate files.

---

## 9. Clustering Stage (MANDATORY)

### Algorithm

* `MiniBatchKMeans`

### k Selection

* Use silhouette score
* Valid k range: `[2, min(20, sqrt(M))]`
* Prefer smaller k if silhouette scores plateau

### Output

* `cluster_labels`
* `cluster_centers`

Clustering MUST run for all `M ≥ 2`.

---

## 10. Distance-to-Centroid Stage (MANDATORY)

For each deduplicated embedding `x_i`:

```python
distance_i = || x_i - centroid[label_i] ||
```

Output:

* `distances` array of shape `(M,)`

This is required for:

* representative comment selection
* extreme/outlier analysis
* LlamaIndex prompts

---

## 11. Resume Logic (STRICTLY LIMITED)

Resume logic applies **only** to:

* embedding batches in `.tmp_*`

Resume logic MUST NOT:

* skip deduplication
* skip clustering
* skip distance computation
* treat existing `.npz` as complete

---

## 12. Logging Requirements (MANDATORY)

The following logs MUST appear in order for each file:

```
🔄 Resuming embeddings (if applicable)
🧹 Deduplicating embeddings
🔍 Starting clustering for X deduplicated embeddings
📊 k=… → silhouette=…
✅ Optimal clusters selected: k=…
📐 Computing distance-to-centroid
💾 Saving final embeddings with clustering results
🧹 Cleaning up temp directory
```

Absence of these logs indicates a pipeline violation.

---

## 13. Failure Semantics

If the pipeline crashes:

* `.tmp_*` remains
* final `.npz` is NOT written
* next run resumes embedding automatically

If final `.npz` exists:

* pipeline is considered complete
* `.tmp_*` must not exist

---

## 14. Non-Goals (Explicit)

❌ No skipping clustering
❌ No partial `.npz` files
❌ No semantic outputs inside `.tmp_*`
❌ No file-based gating of pipeline stages

---

## 15. Acceptance Criteria

This refactor is complete when:

* Clustering always runs after deduplication
* Final `.npz` always contains labels, centroids, and distances
* Temp directories are cleaned after success
* Monitoring loop works reliably for repeated uploads
* No silent skips occur due to existing output files

---

## 16. Architectural Rule (DO NOT VIOLATE)

> **Temporary state is for resuming work.
> Final state is for consuming results.
> Never confuse the two.**

---

