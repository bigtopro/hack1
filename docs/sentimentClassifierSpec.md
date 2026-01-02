
---

# 📄 FINAL SPEC SHEET

## GPU-Only Sentiment Classification Worker (Auto-Monitor Pipeline)

---

## 1. Component Name

**Auto Sentiment Tagger for YouTube Comments (GPU-Only)**

---

## 2. Objective

Build a **standalone Google Colab pipeline** that:

* Monitors a Google Drive folder for uploaded **comments JSON files**
* Automatically runs **emotion classification** on all comments
* Uses a **multilingual Hugging Face emotion model**
* Runs **only in a GPU environment**
* Assigns **top-1 emotion label per comment**
* Saves results to a **dedicated output folder**
* Preserves **comment ID compatibility** with the embedding pipeline

This pipeline must run **independently and in parallel** with the embedding + deduplication pipeline.

---

## 3. Environment Requirements (Strict)

* **GPU REQUIRED**

  * CUDA must be available
  * If GPU is not detected → exit with error
* Optimized for Colab T4 / L4
* CPU fallback is **not allowed**

Reason: this job is designed to be fast and cheap using GPU batching.

---

## 4. Folder Structure (Contract)

### Input Folder (same as embedding pipeline)

```
/content/drive/My Drive/youtubeComments/
```

### Output Folder (new, separate)

```
/content/drive/My Drive/youtubeCommentsSentiments/
```

### Temporary / Internal State

* Optional in-memory tracking only
* No temp folders required on Drive

---

## 5. Input File Format

JSON array. Each element is either:

### Preferred (object format)

```json
{
  "id": "comment_id",
  "comment": "comment text"
}
```

### Fallback (string format)

```json
"comment text"
```

### ID Rules (must match embedding pipeline)

* If `id` exists → use it
* If missing → use array index as ID

---

## 6. Output File Format

### File Naming

```
<input_filename>_sentiments.json
```

### Output Schema (Top-1 Emotion Only)

```json
[
  {
    "id": "comment_id",
    "comment": "original comment text",
    "emotion": "joy"
  }
]
```

### Emotion Label Set (Fixed, Do Not Change)

```
anger
joy
optimism
sadness
```

Exactly one label per comment.

---

## 7. Model Specification

### Hugging Face Model

```
cardiffnlp/twitter-xlm-roberta-base-emotion
```

### Model Properties

* Multilingual
* 4-class emotion classification
* Outputs logits → softmax → argmax

### Output Handling

* Compute softmax
* Select **top-1 emotion**
* Discard other scores (no probability storage required)

---

## 8. Processing Logic (End-to-End Flow)

```
Mount Google Drive
↓
Verify CUDA availability (exit if not available)
↓
Load tokenizer + model (GPU, fp16 if supported)
↓
Monitor input folder for new JSON files
↓
When a new file appears:
    ↓
    Load comments + IDs
    ↓
    Batch emotion classification on GPU
    ↓
    Assign top-1 emotion per comment
    ↓
    Save results to sentiments folder
    ↓
    Mark file as processed (in-memory or filename-based)
↓
Continue monitoring
```

---

## 9. Monitoring Logic (Must Match Embedding Pipeline)

* Periodically scan input folder (e.g., every 10–30 seconds)
* Detect **new JSON files**
* Do **not reprocess files** that already have:

  ```
  <filename>_sentiments.json
  ```

  present in the output folder
* Process files **one at a time**
* Log each lifecycle stage clearly

This logic must mirror the embedding worker’s “wait → detect → process” behavior.

---

## 10. Batching & Performance Requirements

* Use `torch.no_grad()`
* Use GPU tensor operations only
* Configurable batch size (default: 16 or 32)
* Model loaded **once per session**
* Avoid per-comment inference

Target performance:

* ~50k comments in a few minutes on Colab T4

---

## 11. Logging Requirements

Logs must include:

* GPU detection status
* File detection
* Number of comments loaded
* Batch progress
* Output save location
* Completion confirmation

Example:

```
📄 Detected file: abc_comments.json
🧠 Running emotion classification on 11,788 comments (GPU)
💾 Saved results to youtubeCommentsSentiments/abc_comments_sentiments.json
✅ Completed sentiment tagging for abc_comments.json
```

---

## 12. Error Handling Rules

* If GPU not available → **exit immediately**
* If JSON is invalid → log error and skip file
* If output file already exists → skip input file
* Do not crash the monitor loop on single-file failure

---

## 13. Explicit Non-Goals (Out of Scope)

❌ No embeddings
❌ No deduplication
❌ No clustering
❌ No LLM summarisation
❌ No database writes
❌ No CPU fallback

This worker does **emotion tagging only**.

---

## 14. Integration Contract (Why This Works)

* `id` field matches embedding pipeline exactly
* Outputs can be joined later via:

  ```
  comment_id
  ```
* Pipelines can run **fully in parallel**
* Either pipeline can be re-run independently

---

## 15. Design Principle (Lock This In)

> **Embeddings answer “what are people saying?”
> Emotions answer “how do they feel?”
> Separate pipelines. Same IDs. Clean join later.**

---
