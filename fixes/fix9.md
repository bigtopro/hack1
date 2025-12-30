
---

# 🔧 FIXES PROMPT FOR CODING AGENT

## Emotion-First Analysis + Dashboard Alignment

---

## CONTEXT (DO NOT CHANGE LOGIC OUTSIDE THIS SCOPE)

We have an analysis pipeline that:

* Processes ~50k comments
* Uses a sentiment classifier (~27 emotions)
* Uses KMeans clustering (~50 clusters)
* Does **NOT** use deduplication
* Uses an LLM (Groq) **only to explain reasons**, not compute stats

The **product direction is emotion-first storytelling**, not cluster-first.

---

## GOAL OF THIS CHANGE

Refactor the **analysis file only** so that:

1. **Emotions become the primary analysis object**
2. Clusters are used only to **explain reasons behind emotions**
3. LLM outputs are **structured JSON**, not free-text blobs
4. Dashboard JSON matches the new emotion-first layout
5. Dashboard Markdown mirrors the same structure
6. No increase in LLM calls or token usage

---

## REQUIRED FIXES (IMPLEMENT ALL)

---

## FIX 1 — Make emotions the primary output object

### ❌ CURRENT (problem)

```python
results = {
  "clusters": [...],
  "themes": [...],
  "sentiment_overview": ...
}
```

### ✅ REQUIRED

Refactor output so emotions are the top-level object:

```json
{
  "emotions": [
    {
      "emotion": "confusion",
      "percentage": 17.4,
      "confidence": "high",
      "summary": "...",
      "reasons": [...]
    }
  ],
  "selected_clusters": [...],
  "summary_stats": {...}
}
```

Clusters must **not** be top-level unless explicitly promoted.

---

## FIX 2 — Change LLM emotion analysis to STRUCTURED JSON output

### ❌ CURRENT (problem)

LLM returns free-form text and is stored as:

```python
emotion_results[emotion] = response_text
```

This breaks dashboard rendering.

---

### ✅ REQUIRED

Update the emotion-analysis prompt to **require strict JSON output**.

#### REQUIRED OUTPUT FORMAT (LLM must follow exactly):

```json
{
  "emotion": "<emotion>",
  "reasons": [
    {
      "label": "Short reason label",
      "explanation": "2–3 sentence explanation grounded in comments",
      "relative_importance": "high | medium | low"
    }
  ]
}
```

#### Code requirements:

* Parse JSON safely
* If parsing fails, retry once
* If it still fails, store a fallback reason:

```json
{
  "label": "Unclear",
  "explanation": "The comments show mixed reasons without a dominant cause.",
  "relative_importance": "low"
}
```

---

## FIX 3 — Emotion → Cluster selection logic (MANDATORY)

For each emotion:

1. Filter comments by emotion
2. Group by `kmeans_cluster_id`
3. Rank clusters by count **within that emotion**
4. Select **top 3–5 clusters max**
5. Ignore all other clusters

This prevents 50-cluster overload.

---

## FIX 4 — Sampling rules (LOCKED)

For each `(emotion, cluster)`:

* Total samples: **8–12**
* Selection:

  * 70% closest to centroid
  * 30% furthest (edge cases)

Do **not** exceed this.

---

## FIX 5 — Emotion confidence (NO DEDUP)

Since deduplication is removed, define confidence structurally:

```python
if emotion_pct >= 10 and num_clusters <= 3:
    confidence = "high"
elif emotion_pct >= 5:
    confidence = "medium"
else:
    confidence = "low"
```

Store this in dashboard JSON.

---

## FIX 6 — Dashboard JSON restructuring

### ❌ REMOVE / DE-EMPHASIZE

* Cluster-first sections
* Per-cluster long explanations

### ✅ ADD / ENFORCE

Each emotion object must contain:

```json
{
  "emotion": "anger",
  "percentage": 12.4,
  "confidence": "medium",
  "summary": "Short 1–2 sentence LLM summary",
  "reasons": [
    {
      "label": "Audio issues",
      "share_estimate": 39,
      "explanation": "...",
      "sample_comments": {
        "representative": [...],
        "edge_cases": [...]
      }
    }
  ]
}
```

> `share_estimate` is based on cluster sizes within the emotion.

---

## FIX 7 — Dashboard Markdown must mirror JSON

Generate `analysis_dashboard_*.md` **from the JSON**, not from LLM text.

### REQUIRED STRUCTURE

```md
## Emotional Snapshot
- Joy: 34%
- Interest: 28%
- Confusion: 17%

## Emotion: Confusion (High confidence)

### Why viewers feel confused
- Pacing & transitions
- Missing context

### Pacing & transitions (42%)
Representative comments:
- ...
```

No cluster-centric sections allowed.

---

## FIX 8 — Remove legacy / dead logic

Delete or disable:

* Deduplication references
* Per-cluster exhaustive loops
* Cluster-first report sections
* Unused variables (e.g. cluster_weighted_sentiment_dist)

If the code loops over **all clusters**, it is wrong.

---

## FIX 9 — LLM usage constraints (DO NOT VIOLATE)

* Max 1 LLM call per emotion
* Max 8 emotions analyzed
* Max output tokens: 300
* Temperature ≤ 0.2
* Truncate prompts before sending

No additional calls allowed.

---

## ACCEPTANCE CHECKLIST (MUST PASS)

Before marking complete, verify:

* [ ] Dashboard JSON is emotion-first
* [ ] LLM outputs parse as JSON
* [ ] No per-cluster deep dives
* [ ] Markdown dashboard mirrors JSON
* [ ] Total LLM calls ≤ 15 per run
* [ ] Code runs without dedup logic
* [ ] Frontend can render without cluster IDs

---

## FINAL NOTE TO AGENT

Do **not** add new features.
Do **not** optimize further.
Do **not** refactor unrelated files.

This change is about **clarity, structure, and product alignment**, not performance.

---

