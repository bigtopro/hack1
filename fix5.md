
---

# ✅ FINAL MODIFICATION SPEC SHEET

## Analysis → Report → Dashboard JSON → Dashboard Markdown

---

## 0. Objective (What This Change Achieves)

From a **single analysis run**, the system must now produce **three synchronized outputs**:

1. **Full Analysis Report** (deep, narrative)
   → `analysis_report_<timestamp>.md`

2. **Dashboard JSON** (structured, UI-ready)
   → `analysis_dashboard_<timestamp>.json`

3. **Dashboard Markdown Preview** (human-reviewable, mirrors UI)
   → `analysis_dashboard_<timestamp>.md`

All three are generated from the **same computed data**, not re-analysed.

---

## 1. Output Artifacts (Final)

| File                        | Purpose                             | Consumer |
| --------------------------- | ----------------------------------- | -------- |
| `analysis_report_*.md`      | Deep reasoning, demo, documentation | Humans   |
| `analysis_dashboard_*.json` | Canonical analytics payload         | Frontend |
| `analysis_dashboard_*.md`   | Review & QA of dashboard content    | You      |

The **JSON is the source of truth** for the dashboard.
The **dashboard MD is a formatted rendering of that JSON**.

---

## 2. Dashboard JSON (Canonical Contract)

The backend MUST produce this JSON structure:

```json
{
  "meta": { ... },
  "audience_overview": { ... },
  "sentiment_overview": { ... },
  "clusters": [ ... ],
  "themes": [ ... ],
  "opportunities": [ ... ],
  "risks": [ ... ]
}
```

Everything below builds on this.

---

## 3. Confidence Marker Logic (REQUIRED, Structural)

Implement once, reuse everywhere.

```python
def confidence_label(cluster_size, entropy):
    if cluster_size >= 250 and entropy < 1.0:
        return "high"
    if cluster_size >= 120 and entropy < 1.5:
        return "medium"
    return "low"
```

Use this for:

* clusters
* themes
* opportunities
* risks

This builds **trust without probabilities**.

---

## 4. Sampling Logic (FINAL, LOCKED)

### For each cluster:

| Cluster size | Total samples |
| ------------ | ------------- |
| Large        | 12            |
| Medium       | 10            |
| Small        | 8             |

### Selection method:

* **70% closest to centroid** (core signal)
* **30% furthest from centroid** (edge cases)

```python
n_core = int(n_samples * 0.7)
n_edge = n_samples - n_core

core = sorted_by_distance[:n_core]
edge = sorted_by_distance[-n_edge:]
```

Store samples in JSON as:

```json
"sample_comments": {
  "centroid": [...],
  "edge_cases": [...]
}
```

---

## 5. Dashboard JSON – Section Details

### 5.1 Meta

```json
"meta": {
  "video_id": "DWCl2dN6hpg",
  "total_comments": 11788,
  "unique_comments": 983,
  "analysis_timestamp": "2025-12-30T12:13:30Z",
  "model": "llama-3.1-8b-instant"
}
```

---

### 5.2 Audience Overview (Top Chart)

```json
"audience_overview": {
  "cluster_distribution": [
    {
      "cluster_id": 0,
      "label": "Video quality",
      "comment_count": 342,
      "percentage": 29.1,
      "dominant_sentiment": "joy",
      "confidence": "high"
    }
  ]
}
```

---

### 5.3 Sentiment Overview

```json
"sentiment_overview": {
  "distribution": {
    "joy": 34,
    "interest": 28,
    "confusion": 17,
    "anger": 11,
    "other": 10
  },
  "dominant_sentiment": "joy",
  "sentiment_entropy": 1.38
}
```

Computed from **non-deduplicated sentiment JSON**.

---

### 5.4 Clusters (Main Cards)

```json
{
  "cluster_id": 2,
  "label": "Pacing and duration",
  "comment_count": 187,
  "percentage": 15.9,
  "sentiment_distribution": {
    "interest": 41,
    "confusion": 27,
    "joy": 18,
    "anger": 9,
    "other": 5
  },
  "entropy": 1.42,
  "confidence": "medium",
  "summary": "Mixed feedback on pacing, with tension between depth and conciseness.",
  "sample_comments": {
    "centroid": [...],
    "edge_cases": [...]
  }
}
```

---

### 5.5 Themes (LLM, Structured)

```json
{
  "theme": "Advanced examples",
  "why": "Repeated requests indicate unmet learning depth",
  "implication": "Audience is maturing faster than content",
  "recommended_action": "Create advanced follow-up series",
  "confidence": "high"
}
```

---

### 5.6 Opportunities vs Risks

```json
"opportunities": [
  {
    "type": "content_expansion",
    "signal": "High engagement on advanced topics",
    "action": "Launch advanced tutorial series",
    "confidence": "high"
  }
]
```

```json
"risks": [
  {
    "type": "production_quality",
    "signal": "Audio complaints in outdoor segments",
    "impact": "Reduced retention",
    "confidence": "medium"
  }
]
```

---

## 6. NEW: Dashboard Markdown (`analysis_dashboard_*.md`)

### Purpose

* Human-readable
* Mirrors UI
* Generated **from the JSON**
* No LLM calls

---

## 7. Dashboard Markdown Structure (MANDATORY)

```md
# Audience Insights Dashboard

## Overview
- Total comments: 11,788
- Unique comments (deduplicated): 983
- Dominant sentiment: Joy
- Analysis confidence: High

---

## Audience Signals at a Glance
| Theme | % of Comments | Dominant Emotion | Confidence |
|-----|--------------|------------------|------------|
| Video quality | 29.1% | Joy | High |
| Topic depth | 25.3% | Interest | High |
| Pacing | 15.9% | Mixed | Medium |

---

## Discussion Clusters

### Video Quality & Production (High confidence)
**Summary:** Viewers praise visuals and editing quality.

**Core comments:**
- ...
- ...

**Edge cases:**
- ...

---

## Common Ideas & Themes
- High engagement around practical examples
- Requests for advanced follow-up content

---

## Opportunities
- 🚀 Create an advanced tutorial series (High confidence)
- 📈 Add downloadable resources (Medium confidence)

---

## Risks
- ⚠️ Audio quality issues in outdoor segments (Medium confidence)
```

This file should feel like **a readable version of your dashboard**.

---

## 8. Implementation Rule (IMPORTANT)

* JSON is built first
* Markdown dashboard is rendered from JSON
* No duplicated logic

```python
dashboard_json = build_dashboard_json(...)
dashboard_md = render_dashboard_md(dashboard_json)
```

---

## 9. LLM Usage (No Change)

* Same prompts
* Same rate limits
* Same truncation
* No extra calls

This change adds **zero LLM cost**.

---

## 10. What NOT to Do

* ❌ Do not let MD and JSON diverge
* ❌ Do not regenerate text via LLM for dashboard MD
* ❌ Do not add more comments to prompts
* ❌ Do not increase token budgets

---

## 11. Final Outcome

After this spec:

* You can **review dashboard content instantly**
* Frontend has **clean, predictable data**
* Confidence markers build trust
* Sampling quality improves insight depth
* System is product-ready

---

