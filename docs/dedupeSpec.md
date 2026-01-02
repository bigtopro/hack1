Good catch — **no, the previous spec did not explicitly encode “Google Colab–first” constraints or optimizations**. That’s important, and your coding agent *will* miss things if it’s implicit.

Below is a **revised, Colab-aware spec sheet** that you can hand over as-is.
This version **explicitly assumes Google Colab (free tier, T4 GPU)** and bakes in the **necessary performance and memory constraints**.

---

# 📄 SPEC SHEET (REVISED – GOOGLE COLAB OPTIMIZED)

## Semantic Deduplication Layer (Post-Embedding, Pre-Save)

---

## 0. Execution Environment (MANDATORY)

**Target environment:**

* Google Colab (Free tier)
* GPU: NVIDIA T4 (16 GB VRAM, shared)
* Python 3.x
* Limited RAM (~12–13 GB)

⚠️ **All design decisions must be compatible with Colab constraints**:

* No O(N²) memory allocations
* No full similarity matrices
* No FAISS dependency
* No extra disk I/O passes

---

## 1. Objective

Add a **semantic (cosine-similarity) deduplication step** to the existing embedding pipeline that:

* Runs **inside Google Colab**
* Executes **in-memory**
* Runs **after embeddings are generated**
* Runs **before saving `.npz` files to Google Drive**
* Produces **only deduplicated embeddings as final output**

---

## 2. Why This Is Needed (Context for Agent)

* Byte-level deduplication **does not work** for embeddings
* Embeddings differ numerically even for identical text (FP16, GPU)
* Deduplication must be **similarity-based**
* Running dedup **after saving** wastes Colab I/O and memory

This layer improves:

* clustering quality
* runtime
* memory usage
* downstream LLM analysis

---

## 3. Placement in Pipeline (STRICT)

### Existing Flow

```
JSON → Embedding (batched) → Save NPZ
```

### Required Flow

```
JSON
→ Embedding (batched, GPU)
→ Concatenate embeddings
→ 🔹 Cosine deduplication (CPU, in-memory)
→ Save deduplicated NPZ
```

❌ Deduplication must NOT:

* run after saving
* read files back from Drive
* create separate deduplicated folders

---

## 4. Deduplication Definition

### Inputs (in memory)

* `embeddings`: NumPy array `(N, D)`

  * L2-normalized
  * float16 or float32
* `comment_ids`: list of length `N`
* `comments`: list of length `N`

### Outputs

* `dedup_embeddings`: NumPy array `(M, D)`
* `dedup_comment_ids`: list of length `M`
* `dedup_comments`: list of length `M`

Where `M ≤ N`.

---

## 5. Similarity Rule (Exact)

Because embeddings are L2-normalized:

```
cosine_similarity = dot_product
```

For each embedding `E_i`:

> Keep `E_i` **only if**
> `max(dot(E_i, E_kept)) < SIMILARITY_THRESHOLD`

### Default Threshold (Colab-safe)

```python
SIMILARITY_THRESHOLD = 0.995
```

This removes:

* copy-paste spam
* repeated boilerplate comments
* near-identical phrasing

without collapsing real opinions.

---

## 6. Algorithm Constraints (Colab Optimization Rules)

The deduplication algorithm MUST:

* Be **single-pass**
* Be **order-preserving**
* Avoid NxN similarity matrices
* Avoid large temporary allocations
* Run in CPU RAM (not GPU)
* Work for ~10k–100k embeddings

### Explicitly Forbidden

❌ `cosine_similarity(embeddings, embeddings)`
❌ Full similarity matrix
❌ FAISS (overkill + setup cost)
❌ Disk-based dedup passes

---

## 7. Required Function (New, Mandatory)

Agent must implement:

```python
def cosine_deduplicate(
    embeddings: np.ndarray,
    ids: list,
    texts: list,
    similarity_threshold: float = 0.995
) -> tuple[np.ndarray, list, list]:
    """
    Colab-optimized cosine-based deduplication.
    Assumes embeddings are L2-normalized.
    """
```

### Functional Requirements

* Iterate embeddings sequentially
* Compare against **already-kept embeddings only**
* Use dot product for similarity
* Maintain alignment between:

  * embedding
  * ID
  * comment text

---

## 8. Integration Point (Exact Code Location)

Inside:

```
CommentEmbedder.process_file()
```

### Insert **after**:

```python
embeddings_array = np.vstack(all_embeddings)
```

### Insert **before**:

```python
np.savez_compressed(...)
```

This is **non-negotiable**.

---

## 9. Logging (Required for Colab Debugging)

Add logs:

```text
🧹 Running cosine-based deduplication (threshold=0.995)...
🧹 Deduplication reduced comments from {N} → {M}
```

This helps:

* demo clarity
* performance debugging
* sanity checking in Colab

---

## 10. Saving Behavior (Unchanged)

Only deduplicated data is saved:

```python
np.savez_compressed(
    output_path,
    embeddings=dedup_embeddings,
    ids=dedup_comment_ids
)
```

No additional files, metadata, or folders.

---

## 11. Performance Expectations (Colab Free Tier)

Target performance for ~10k comments:

* Dedup runtime: **seconds, not minutes**
* Memory usage: stays within Colab RAM
* No GPU memory impact
* No additional Drive reads/writes

---

## 12. Non-Goals (Explicit)

❌ Text-hash deduplication
❌ Exact float equality checks
❌ Post-save deduplication
❌ LLM-based deduplication
❌ Clustering changes

This spec covers **deduplication only**.

---

## 13. Design Principle (For the Agent)

> **In Colab, geometry beats brute force.**
> Deduplicate embeddings by **distance**, not equality.

---

## 14. Acceptance Criteria

This task is complete when:

* Output `.npz` files contain **fewer rows than input comments**
* Deduplication runs **before saving**
* Pipeline runs end-to-end in Google Colab
* No memory errors occur
* Clustering input size is reduced automatically

---

If you want, next I can:

* compress this into a **Qwen-Coder CLI instruction prompt**
* write a **minimal test harness** to validate dedup works
* add **adaptive thresholds** based on dataset size
* or make a **follow-up spec** for K-means + cluster analysis

You’re doing this the *right* way: correctness first, scale second, demos last.
