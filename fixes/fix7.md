Perfect. Below is a **handoff-ready, detailed modification spec sheet** you can give directly to your coding agent.
It is written to be **unambiguous, prescriptive, and minimal-change**, based on the **new pipeline reality**:

* ❌ No deduplication upstream
* ✅ √N clustering (k ≈ 50)
* ✅ Frequency = signal
* ✅ Analysis engine = reasoning, not geometry

---

# 📄 MODIFICATION SPEC SHEET

## Analysis Engine Alignment (Post-Dedup Removal & √N Clustering)

### Component

**Local Analysis Engine (Embeddings × Emotions × LLM Reasoning)**

---

## 1. Objective of This Change

Update the **analysis engine** to correctly interpret data produced by the **new clustering pipeline**, where:

* No embedding-level deduplication exists
* Repetition and volume are **intentional signal**
* Clusters are **coarse semantic buckets**
* Fine-grained meaning comes from:

  * emotions
  * representative sampling
  * LLM reasoning

This change is **conceptual cleanup**, not a rewrite.

---

## 2. Key Design Decisions (LOCKED)

These assumptions must be reflected everywhere in the analysis engine:

1. **Deduplication does not exist**
2. **Cluster size = importance signal**
3. **Semantic diversity ≠ deduplicated count**
4. **LLMs handle repetition better than cosine thresholds**
5. **Clustering is an indexing layer, not an optimization layer**

Any logic contradicting the above must be removed.

---

## 3. REMOVE All Deduplication-Related Concepts (CRITICAL)

### ❌ Functions to DELETE entirely

Remove these functions if present:

* `_calculate_deduplication_impact`
* `_calculate_cluster_deduplication_signals`
* Any helper that:

  * compares “raw vs deduped”
  * computes “compression ratio”
  * references “semantic deduplication”

These concepts are **no longer valid**.

---

### ❌ Variables / Fields to REMOVE

Remove all usage of fields like:

* `deduplicated_comments`
* `semantic_ids`
* `compression_ratio`
* `deduplication_savings`
* `unique_comment_count`

They must not appear in:

* dashboard JSON
* markdown reports
* logs
* LLM prompts

---

## 4. Rename & Reframe Sentiment Metrics (IMPORTANT)

The engine currently distinguishes between:

* “raw sentiment”
* “semantic / deduplicated sentiment”

This framing is now **incorrect**.

---

### ✅ New Correct Framing

#### 4.1 Raw Sentiment (KEEP, UNCHANGED)

Definition:

> Emotion distribution over **all comments**

Implementation:

```python
raw_sentiment_distribution = Counter(comment_emotions)
```

Interpretation:

* Emotional **volume**
* How loud each emotion is

This stays exactly as-is.

---

#### 4.2 Semantic / Dedup Sentiment (REPLACE)

❌ Old meaning:

> sentiment after deduplication

✅ New meaning:

> **cluster-weighted sentiment** (emotion diversity across topics)

---

### ✅ REQUIRED IMPLEMENTATION CHANGE

Replace any “semantic sentiment” calculation with:

```python
cluster_weighted_sentiment = Counter()

for cluster_id in clusters:
    dominant_emotion = most_common_emotion_in_cluster(cluster_id)
    cluster_weighted_sentiment[dominant_emotion] += 1

normalize(cluster_weighted_sentiment)
```

Interpretation:

* “How many distinct topics express anger?”
* Not “how many comments are angry?”

This is **far more meaningful**.

---

### ✅ Required Renaming (Variables & Output)

Rename everywhere:

| Old Name                          | New Name                                  |
| --------------------------------- | ----------------------------------------- |
| `semantic_sentiment_distribution` | `cluster_weighted_sentiment_distribution` |
| `semantic_sentiment`              | `topic_level_sentiment`                   |
| `deduplicated_sentiment`          | ❌ remove                                  |

---

## 5. KEEP Representative Sampling (DO NOT CHANGE)

This logic is **correct and essential**:

* Sampling comments closest to cluster centroid
* Using these for:

  * summaries
  * LLM prompts
  * explanation generation

This is now your **effective deduplication** mechanism.

❗ Do not:

* reduce sample size further
* dedup within samples
* weight samples artificially

---

## 6. Update LLM Prompt Language (TEXT-ONLY CHANGE)

Your prompt logic is strong, but wording must change.

---

### ❌ REMOVE phrases like:

* “after deduplication”
* “compression”
* “semantic deduplicated comments”
* “unique comments only”

---

### ✅ REPLACE with:

* “semantic diversity”
* “topic-level patterns”
* “distinct reasons”
* “emotion expressed across topics”

No prompt structure change needed — **only language**.

---

## 7. Dashboard & Report Cleanup

### ❌ REMOVE sections referencing:

* deduplication impact
* compression savings
* raw vs dedup ratios

---

### ✅ ADD / KEEP sections like:

* Emotion Volume (raw sentiment)
* Emotion Diversity (cluster-weighted sentiment)
* Top clusters by emotional intensity
* Emotion × Cluster breakdown

These align with the new pipeline.

---

## 8. What NOT to Change (Explicit)

Do NOT:

* Reintroduce cosine deduplication
* Add silhouette scores
* Add cluster purity metrics
* Normalize away frequency
* Reduce cluster count for “cleanliness”

These all conflict with the intended architecture.

---

## 9. Acceptance Criteria (Very Important)

The modification is complete when:

* No deduplication language exists anywhere
* No code references “dedup”, “semantic ids”, or “compression”
* Raw sentiment = volume
* Cluster-weighted sentiment = diversity
* Reports read **cleanly and intuitively**
* LLM outputs focus on **reasons**, not mechanics

---

## 10. Conceptual Rule for the Agent (Read This)

> **Repetition is signal, not noise.
> Clusters define topics.
> Emotions define tone.
> LLMs explain reasons.**

The analysis engine must reinforce this model — not fight it.

---

## Final Note

This spec intentionally:

* avoids refactors
* avoids algorithm changes
* focuses on **correct interpretation**

Once applied, your system becomes:

* fast (no O(N²))
* emotionally rich
* topic-aware
* LLM-aligned
* analytically honest

