🔧 FEATURE SPEC SHEET
Basic Summary Stats + Deduplication Impact Layer
1. Goal of This Feature

Add interpretable, numeric grounding to the analysis so users can answer:

How do viewers feel overall?

Is this emotion widespread or just repetitive?

Which clusters are emotionally consistent vs noisy?

What changed after deduplication, and why does that matter?

This data must be:

Machine-readable (JSON)

Human-readable (Dashboard MD)

Frontend-friendly (percentages, deltas, labels)

2. New JSON Section: summary_stats
Add to top-level dashboard JSON
"summary_stats": {
  "overall_sentiment": { ... },
  "deduplication_impact": { ... },
  "cluster_sentiment_stats": [ ... ]
}


This section feeds:

dashboard header widgets

explanation tooltips

trust signals

3. Overall Sentiment Breakdown (Percentages)
Source

Raw sentiment → all sentiment JSON

Semantic sentiment → deduplicated IDs only

JSON structure
"overall_sentiment": {
  "raw": {
    "joy": 34.2,
    "interest": 28.1,
    "confusion": 17.4,
    "anger": 11.0,
    "other": 9.3
  },
  "semantic": {
    "joy": 26.8,
    "interest": 31.5,
    "confusion": 22.4,
    "anger": 12.1,
    "other": 7.2
  },
  "dominant_raw": "joy",
  "dominant_semantic": "interest"
}

Why this matters

Raw = emotional volume

Semantic = emotional diversity

Differences tell a story by themselves

4. Deduplication Impact (Critical Trust Builder)
Compute

For each emotion:

delta = semantic_pct - raw_pct

JSON structure
"deduplication_impact": {
  "compression_ratio": 0.083,
  "emotion_shifts": [
    {
      "emotion": "joy",
      "raw_pct": 34.2,
      "semantic_pct": 26.8,
      "delta": -7.4,
      "interpretation": "High repetition of similar positive reactions"
    },
    {
      "emotion": "confusion",
      "raw_pct": 17.4,
      "semantic_pct": 22.4,
      "delta": +5.0,
      "interpretation": "Fewer comments, but many distinct confusion reasons"
    }
  ]
}

Interpretation rules (deterministic, no LLM)

Semantic ↓ vs Raw → repetition / echoing

Semantic ↑ vs Raw → diverse, structurally important

Large deltas (>±4%) → highlight-worthy

5. Per-Cluster Sentiment Breakdown
Add to each cluster object
{
  "cluster_id": 2,
  "label": "Pacing and duration",
  "comment_count": 187,
  "sentiment_breakdown": {
    "interest": 41,
    "confusion": 27,
    "joy": 18,
    "anger": 9,
    "other": 5
  },
  "dominant_sentiment": "interest",
  "sentiment_entropy": 1.42,
  "confidence": "medium"
}

Notes

Percentages must sum to ~100

Use semantic data only

This powers:

stacked bars

cluster emotion badges

hover tooltips

6. Cluster-Level Deduplication Insight (Optional but Powerful)
Derived metric
cluster_raw_mentions / cluster_unique_comments

JSON
"deduplication_signal": {
  "cluster_id": 2,
  "raw_mentions": 612,
  "unique_comments": 187,
  "repetition_factor": 3.27,
  "interpretation": "Many viewers repeat similar pacing concerns"
}

Display use

“Echo chamber” indicator

Helps explain why some issues feel louder

7. Dashboard Markdown Additions
Add section near top
## Sentiment Snapshot

**Overall viewer emotions (raw):**
- Joy: 34.2%
- Interest: 28.1%
- Confusion: 17.4%
- Anger: 11.0%

**After semantic deduplication:**
- Interest becomes dominant (31.5%)
- Confusion increases (+5.0%), indicating diverse reasons
- Joy decreases (−7.4%), indicating repetition

**What this means:**
Raw sentiment reflects volume.
Semantic sentiment reflects diversity of reasons.

Add per-cluster snippet
**Emotional makeup:**
- Interest (41%)
- Confusion (27%)
- Joy (18%)

**Interpretation:**  
This topic attracts curiosity but also unresolved questions.

8. UI Elements This Unlocks

You can now safely add:

📊 Raw vs Semantic sentiment toggle

🔁 Deduplication impact tooltip

🧠 “Emotion diversity” badges

⚠️ High-repetition warning on clusters

🧪 Confidence labels users can trust

9. What NOT to Do

❌ Do not LLM-generate percentages

❌ Do not hide deduplication effects

❌ Do not collapse raw + semantic into one number

❌ Do not over-explain — let numbers speak first