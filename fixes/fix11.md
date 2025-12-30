Got it. Below is a **clean, handoff-ready FIX SPEC SHEET** that your coding agent can implement without interpretation.
This consolidates **only the required and recommended fixes** from the last review — no redesign, no new features.

You can paste this **as-is** into your agent.

---

# 🔧 FIX SPEC SHEET

## Emotion-First Analysis – Correctness & UX Alignment Fixes

---

## SCOPE (IMPORTANT)

* **File scope**: analysis file only
* **Do NOT** change:

  * clustering logic
  * sentiment classifier
  * sampling rules
  * Groq API usage
  * dashboard layout design
* This spec is **cleanup + correctness**, not a redesign

---

## FIX 1 — Correct cluster size & percentages (CRITICAL BUG)

### ❌ Problem

Cluster sizes are computed from **distributions**, not actual counts:

```python
cluster_size = sum(cluster_info.values())
```

This is incorrect and breaks:

* `comment_count`
* `percentage`
* confidence logic

---

### ✅ Required Fix

Cluster size must be computed from **raw counts**, not percentages.

#### Correct approach

Use the cluster → comments mapping:

```python
cluster_size = len(cluster_sentiment_map.get(cluster_id, []))
```

Then compute percentages as:

```python
percentage = (cluster_size / total_comments) * 100
```

---

### Acceptance check

* `comment_count` is an integer
* Percentages across clusters sum correctly
* Confidence labels change appropriately after fix

---

## FIX 2 — Remove / rename misleading `unique_comments` field (CRITICAL)

### ❌ Problem

Deduplication is removed, but dashboard metadata still shows:

```python
"unique_comments": len(self.id_to_index)
```

This is misleading.

---

### ✅ Required Fix (pick ONE)

#### Option A — Rename (recommended)

```json
"embedded_comments": <count>
```

#### Option B — Remove entirely

Delete the field from:

* dashboard JSON
* dashboard MD
* meta section

---

### Acceptance check

* No field labeled “unique” unless dedup exists
* Meta numbers are interpretable by users

---

## FIX 3 — Fix broken `themes` / emotion summary block (CRITICAL BUG)

### ❌ Problem

You are slicing a **dict** as if it were text:

```python
"why": analysis[:200]
```

This is invalid and can silently fail.

---

### ✅ Required Fix

Either **remove `themes` completely** (preferred),
OR extract from structured reasons:

```python
why = ", ".join(
    reason["label"] for reason in analysis.get("reasons", [])[:3]
)
```

Then store:

```json
{
  "theme": emotion,
  "why": why
}
```

---

### Acceptance check

* No string slicing on dicts
* No runtime errors in this section
* Output is human-readable

---

## FIX 4 — Emotion summaries too weak (UX FIX)

### ❌ Problem

Current summaries are mechanical:

```python
"Key reasons: pacing, audio"
```

This feels robotic in the UI.

---

### ✅ Required Fix

Generate a short narrative summary:

```python
emotion_data["summary"] = (
    f"Viewers mainly feel {emotion} due to "
    f"{', '.join(reason_labels[:2])}."
)
```

No extra LLM calls required.

---

### Acceptance check

* Emotion cards read naturally
* Summary fits in 1–2 lines
* No repetition of reason list formatting

---

## FIX 5 — Sort emotions by importance (UX + ANALYTICS)

### ❌ Problem

Emotions appear in random order (dict iteration).

---

### ✅ Required Fix

Before building the dashboard array:

```python
sorted_emotions = sorted(
    raw_sentiment_dist.items(),
    key=lambda x: x[1],
    reverse=True
)
```

Use `sorted_emotions` everywhere.

---

### Acceptance check

* Top emotion always appears first
* Dashboard order matches visual charts

---

## FIX 6 — Hide cluster IDs from user-facing outputs (UX FIX)

### ❌ Problem

Cluster IDs still appear in:

* dashboard JSON labels
* dashboard markdown headings
* actionable insight context

Example:

```python
"label": f"Cluster {cluster_id}"
```

---

### ✅ Required Fix

Never expose cluster IDs to users.

Replace with:

```python
analysis.get("topic_label", "Topic")
```

Cluster IDs may remain **internally only**.

---

### Acceptance check

* No “Cluster 17” visible in dashboard or MD
* Topics have human-readable names

---

## FIX 7 — Improve emotion confidence logic using entropy (OPTIONAL BUT RECOMMENDED)

### ❌ Problem

Confidence currently ignores fragmentation.

---

### ✅ Recommended Fix

Incorporate entropy:

```python
if emotion_pct >= 10 and num_clusters <= 3 and entropy < 1.2:
    confidence = "high"
elif emotion_pct >= 5:
    confidence = "medium"
else:
    confidence = "low"
```

This prevents false “high confidence” labels.

---

## FIX 8 — Simplify actionable-insights input (QUALITY FIX)

### ❌ Problem

Actionable insights are still cluster-heavy:

```python
Topic {cluster_id}: summary...
```

---

### ✅ Recommended Fix

Build insight context from:

* emotion names
* reason labels
* emotion confidence

Example input:

```text
Anger: audio issues, interruptions
Confusion: pacing, missing context
```

Clusters should **support**, not dominate.

---

## CLEANUP (SAFE TO DO)

You may safely remove:

* unused clustering imports
* silhouette / linkage code
* dead variables related to dedup

This is optional but reduces mental load.

---

## FINAL ACCEPTANCE CHECKLIST

Before marking complete:

* [ ] Cluster sizes use counts, not percentages
* [ ] No dedup language remains
* [ ] Emotion summaries read naturally
* [ ] Emotions sorted by importance
* [ ] No cluster IDs in UI outputs
* [ ] `themes` block fixed or removed
* [ ] Dashboard JSON renders emotion-first
* [ ] Dashboard MD mirrors JSON

---

## FINAL NOTE TO AGENT

Do **not**:

* add new features
* redesign prompts
* refactor architecture
* increase LLM usage

This spec is about **correctness, clarity, and trust** — nothing more.

---

If you want next, I can:

* provide a **diff-style patch** for the top 3 fixes
* sanity-check one final JSON output
* or help you freeze a `v1` tag with confidence
