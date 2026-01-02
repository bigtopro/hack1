Perfect. This is now a **proper “analysis & insight engine”**, and writing a clean spec sheet here is exactly the right move.

Below is a **handoff-ready, end-to-end SPEC SHEET** for the **Analysis Layer**, incorporating **all approved ideas**, explicitly **excluding time-awareness**, and aligned with:

* local Mac execution
* file monitoring in `/Users/venuvamsi/Downloads`
* LlamaIndex + **OpenRouter free LLM**
* your existing `.npz` + sentiment JSON contracts

You can give this directly to your coding agent.

---

# 📄 FINAL SPEC SHEET

## Comment Analysis & Insight Generation Layer

*(Post-Clustering + Post-Sentiment)*

---

## 1. Component Name

**Comment Insight & Audience Analysis Engine**

---

## 2. Objective

Build a **local analysis pipeline** that:

* Monitors a local folder for **two completed output files**:

  * clustered embeddings (`.npz`)
  * sentiment labels (`.json`)
* Joins them safely using **comment IDs**, respecting **deduplication**
* Performs **multi-angle analysis** combining:

  * semantics (clusters)
  * emotions (sentiment labels)
* Uses an LLM (via **LlamaIndex + OpenRouter free model**) to generate:

  * structured insights
  * explanations
  * actionable recommendations
* Produces a **final insight report** suitable for creators / product teams

---

## 3. Environment & Execution Context

### Execution Environment

* Local machine (macOS)
* Python script / notebook
* No GPU required (analysis + LLM only)

### Input Folder (Monitored)

```
/Users/venuvamsi/Downloads
```

The analysis should **start automatically** once the required files are detected.

---

## 4. Input Files & Data Contracts

### 4.1 Cluster File (NPZ)

Loaded from `.npz` file containing:

```python
[
  "embeddings",   # shape (N, D)
  "ids",          # deduplicated comment IDs
  "labels",       # k-means cluster id per embedding
  "centroids",    # cluster centroids
  "distances"     # distance of each embedding to its centroid
]
```

Important:

* `ids` represent **deduplicated semantic comments**
* Every analysis that references clusters must operate on this ID set

---

### 4.2 Sentiment File (JSON)

JSON array of objects:

```json
{
  "id": 5,
  "comment": "Am Want Mark Am want Mark",
  "emotion": "desire"
}
```

Properties:

* Contains **all original comments**
* May include duplicates that were removed during embedding

---

## 5. File Detection Logic

The pipeline should:

1. Continuously monitor `/Users/venuvamsi/Downloads`
2. Detect:

   * one `.npz` clustering file
   * one `_sentiments.json` file
3. Start analysis **only when both are present**
4. Avoid reprocessing the same pair twice (simple filename lock or marker file)

---

## 6. Joining Strategy (CRITICAL)

### 6.1 Sentiment Lookup Table

Create:

```python
sentiment_by_id = { id → emotion }
```

Built from **sentiment JSON**.

---

### 6.2 Deduplication-Aware Join

* Only IDs present in `.npz["ids"]` are used for:

  * cluster-based analysis
  * LLM reasoning
* Sentiment is attached by lookup:

```python
emotion = sentiment_by_id[id]
```

This ensures:

* cluster analysis reflects **unique semantic ideas**
* sentiment is correctly aligned

---

### 6.3 Dual Sentiment Views (Important Insight)

Maintain **two parallel sentiment summaries**:

1. **Raw Viewer Sentiment**

   * Computed from *all* sentiment JSON entries
   * Reflects **volume of emotion**

2. **Semantic Sentiment**

   * Computed only on **deduplicated IDs**
   * Reflects **variety of emotional reasons**

Both must be reported.

---

## 7. Analysis Steps (Core Logic)

### 🔹 Step 1: Global Sentiment Distribution

Compute:

* Percentage of each emotion across:

  * all comments (raw)
  * deduplicated comments (semantic)

Output:

* emotion → percentage table
* short LLM summary:

  * “Overall audience mood”
  * “Dominant vs niche emotions”

---

### 🔹 Step 2: Sentiment → Cluster → Reason Analysis

For **each emotion**:

1. Filter deduplicated comments with that emotion
2. Group them by **k-means cluster ID**
3. Rank clusters by:

   * number of comments
   * average centroid distance (representativeness)
4. Sample **5–10 comments per cluster**
5. Send to LLM via LlamaIndex with prompt:

**Goal**:

> Explain *why* viewers feel this emotion, broken down by distinct reasons.

Output:

* Emotion-specific sections, e.g.:

  * “Reasons for Anger”
  * “Sources of Sadness”
  * “What Drives Desire / Optimism”

---

### 🔹 Step 3: Cluster → Sentiment Mix Analysis

For **each k-means cluster**:

1. Compute emotion distribution inside the cluster
2. Identify:

   * dominant emotion
   * minority emotions
3. Sample representative comments
4. Send to LLM with prompt:

**Goal**:

> Explain what this topic is about and how it emotionally affects viewers.

This answers:

* “What emotions does this topic provoke?”
* “Is this cluster polarizing or consistent?”

---

## 8. Advanced Metrics to Include

### ⭐ A. Sentiment Entropy (Per Cluster)

Compute entropy of emotion distribution inside each cluster.

Interpretation:

* Low entropy → emotionally consistent topic
* High entropy → polarizing / divisive topic

Include this signal in LLM prompts.

---

### ⭐ B. Distance-Weighted Sentiment Influence

When sampling comments:

* Prefer comments **closer to centroid**
* Down-weight far outliers

Purpose:

* Prevent fringe opinions from skewing insights
* Focus LLM reasoning on core ideas

---

## 9. LLM Layer (LlamaIndex + OpenRouter)

### LLM Access

* Use **OpenRouter free model**
* Accessed via **LlamaIndex**
* No fine-tuning required

### LLM Responsibilities

* Summarize clusters
* Explain emotional causes
* Generate actionable insights
* Synthesize across analyses

LLM must **not**:

* recompute statistics
* hallucinate numbers
* override computed metrics

---

## 10. Final Report Structure (Output)

The analysis engine must produce a structured report with sections:

1. **Overall Viewer Sentiment**

   * Raw vs semantic comparison
2. **Key Emotional Drivers**

   * Reasons behind anger, joy, sadness, desire, etc.
3. **Topic-Level Emotional Impact**

   * Cluster summaries with emotion mixes
4. **Engagement Diagnostics**

   * Polarizing vs stable topics
5. **Actionable Insights**

   * Content improvements
   * New video ideas
   * Risk signals
6. **What to Do Next**

   * 3 recommended actions
   * 2 things to keep unchanged
   * 1 experiment worth trying

Output can be:

* Markdown
* JSON + Markdown
* or rendered text report

---

## 11. Explicitly Excluded

❌ Time-based analysis
❌ Trend over time
❌ Re-embedding or reclustering
❌ Re-sentiment classification
❌ Model training

---

## 12. Design Principle (Lock This In)

> **Clusters explain what people talk about.
> Emotions explain how they feel.
> LLMs explain why.**

This layer exists to **connect those three**.

---

## Final Verdict

This analysis layer is:

* conceptually sound
* hard to fake
* difficult to replicate without your pipeline
* genuinely valuable to creators

Once this is built, you’re no longer “analyzing comments” —
you’re **extracting audience psychology**.

Excellent ask. This is the **missing piece** that turns a good analysis layer into a *repeatable, controllable* insight engine.

Below is an **extension to the spec sheet** you already have, adding **explicit system prompts** for **each LLM call**, written so your coding agent can plug them directly into **LlamaIndex + OpenRouter**.

I’ll structure this as:

1. **Global system rules** (used for all calls)
2. **Per-analysis system prompts** (one per LLM task)
3. **Why each prompt is shaped this way** (so you don’t accidentally break it later)

No fluff, no vague “analyze this” prompts.

---

# 🔧 ADDENDUM TO SPEC SHEET

## LLM System Prompts for Comment Analysis Engine

---

## 1️⃣ Global System Prompt (Used for ALL LLM Calls)

This should be passed as the **system prompt** every time.

```
You are an analytical assistant specializing in audience psychology and content analysis.

You must:
- Base your analysis ONLY on the data provided.
- Never invent statistics, percentages, or trends.
- Never assume time-based changes unless explicitly stated.
- Focus on explaining causes, patterns, and implications.
- Be concise, structured, and concrete.
- Avoid generic advice and clichés.

You are not a chatbot.
You are producing insights for creators and analysts who will act on your output.
```

This prompt:

* suppresses hallucination
* discourages “motivational speaker” tone
* forces evidence-based reasoning

---

## 2️⃣ Prompt A — Overall Viewer Sentiment Summary

### Used for:

* Explaining **global sentiment percentages**
* Raw vs deduplicated comparison

### Inputs:

* emotion → percentage (raw)
* emotion → percentage (deduplicated)

### System Prompt

```
You are analyzing aggregated viewer sentiment data.

Your task:
- Interpret the overall emotional state of the audience.
- Compare raw sentiment percentages with deduplicated (semantic) sentiment percentages.
- Explain what this difference implies about emotional intensity vs emotional diversity.

Rules:
- Do NOT restate the numbers verbatim.
- Do NOT speculate beyond the data.
- Focus on what dominates, what is niche, and what is structurally important.

Output format:
- 1 short paragraph: overall emotional tone
- 3–5 bullet points: key takeaways
```

### What this produces

* Not just “most people are happy”
* But: *“joy is high but concentrated, anger is lower but diverse”*

---

## 3️⃣ Prompt B — Emotion → Cluster → Reason Analysis

*(“Why are viewers angry / sad / joyful?”)*

### Used for:

* Each emotion separately
* Clustered explanations

### Inputs:

For a **single emotion**:

* emotion label
* list of clusters:

  * cluster size
  * representativeness score
  * 5–10 sample comments per cluster

### System Prompt

```
You are analyzing viewer comments that all express the same emotion: {EMOTION}.

The comments are grouped into distinct semantic clusters.
Each cluster represents a different underlying reason for this emotion.

Your task:
- Identify the main reason behind each cluster.
- Explain how these reasons differ from one another.
- Focus on causes, not solutions.

Rules:
- Do NOT merge clusters unless they are clearly the same reason.
- Do NOT generalize across clusters.
- Treat each cluster as a separate emotional driver.

Output format:
- Section title: "Reasons for {EMOTION}"
- For each cluster:
  - Short label for the reason
  - 2–3 sentence explanation grounded in the comments
```

### What this produces

Instead of:

> “People are angry because of content quality”

You get:

* “Anger due to misinformation”
* “Anger due to pacing”
* “Anger due to ideological disagreement”

---

## 4️⃣ Prompt C — Cluster → Sentiment Mix Analysis

*(“What emotions does this topic trigger?”)*

### Used for:

* Each **k-means cluster**
* With emotion distribution + samples

### Inputs:

* cluster summary
* emotion percentages within cluster
* sentiment entropy score
* representative comments

### System Prompt

```
You are analyzing a single discussion topic derived from viewer comments.

This topic has:
- A known emotional distribution
- A measurable level of emotional consistency or polarization

Your task:
- Explain what this topic is about.
- Describe how viewers emotionally respond to it.
- Interpret whether the topic is emotionally consistent or emotionally mixed.

Rules:
- Explicitly connect emotions to aspects of the topic.
- If emotions are mixed, explain why.
- Do NOT make recommendations.

Output format:
- Topic summary (2–3 sentences)
- Emotional interpretation (2–3 sentences)
```

### What this produces

* Clear topic summaries
* Emotional framing (“this topic excites some, frustrates others”)

---

## 5️⃣ Prompt D — Engagement & Risk Diagnostics

### Used for:

* Interpreting sentiment entropy
* Identifying polarizing vs stable topics

### Inputs:

* cluster sizes
* entropy scores
* dominant emotions

### System Prompt

```
You are evaluating viewer engagement quality and risk signals.

Your task:
- Identify which topics are emotionally stable.
- Identify which topics are emotionally polarizing.
- Highlight potential engagement risks or strengths.

Rules:
- Do NOT assume growth or decline.
- Focus on emotional structure, not popularity.
- Avoid alarmist language.

Output format:
- 3–5 bullet points:
  - What is emotionally stable
  - What is emotionally polarizing
  - Why this matters for engagement
```

---

## 6️⃣ Prompt E — Actionable Insight Synthesis (Final Section)

### Used for:

* Final report recommendations
* High-level synthesis

### Inputs:

* All prior LLM outputs
* Key statistics (already computed)

### System Prompt

```
You are synthesizing insights from a full viewer comment analysis.

Your task:
- Translate emotional and semantic patterns into actionable insights.
- Focus on content strategy, communication, and audience understanding.

Rules:
- Base every recommendation on earlier analysis.
- Avoid generic advice.
- Be specific and concrete.

Output format:
- 3 recommended actions
- 2 things that should not be changed
- 1 high-risk / high-reward experiment idea
```

### What this produces

* Creator-ready guidance
* Not “post more consistently” nonsense

---

## 7️⃣ Important LLM Usage Rules (For the Agent)

* Every LLM call must:

  * pass **computed statistics explicitly**
  * never ask the model to “calculate”
* LLM outputs are **explanations, not data sources**
* If numbers are missing → model must not guess

---

## Final Design Principle (Very Important)

> **Statistics decide *what*.
> Clusters decide *where*.
> Emotions decide *how*.
> LLMs decide *why*.**

Your prompts enforce that separation.

---

