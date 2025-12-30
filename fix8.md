Perfect — this is the moment where things *snap into place*.
You’re not asking for tweaks anymore; you’re **changing the analysis contract**. So I’ll give you a **final, authoritative spec sheet** that your coding agent can follow to refactor the **analysis file only**, without destabilizing the rest of the pipeline.

I’ll explicitly combine:

* your **emotion → cluster → sample → LLM** idea
* the need to **avoid 50-cluster overload**
* the removal of **deduplication**
* and the reality of **50k comments + Groq limits**

---

# ✅ FINAL SPEC SHEET

## Emotion-First, Cluster-Explained Analysis Architecture

*(Post-Dedup Removal, 50-Cluster Scale)*

---

## 0. What Changed (Design Truths)

These are now **non-negotiable assumptions**:

* ~50 KMeans clusters exist
* Deduplication is **removed**
* Emotion classifier outputs ~27 emotions
* Clusters are **too many to analyze exhaustively**
* Emotions are the **primary narrative unit**
* LLM must explain **reasons**, not summarize everything

Everything below enforces these constraints.

---

## 1. Core Architectural Shift

### OLD (Cluster-First – ❌ does not scale)

```
Cluster → Sentiment → LLM → Report
```

### NEW (Emotion-First – ✅ scales cleanly)

```
Emotion
  └── clusters that express this emotion
        └── representative comments
              └── LLM explains reasons
```

> **Emotion answers “what”**
> **Clusters explain “why”**

---

## 2. High-Level Analysis Flow (Final)

The analysis file must follow **exactly this order**:

### Step 1 — Global Emotion Statistics (No LLM)

* Count emotion distribution across all comments
* Compute percentages
* Rank emotions by volume
* Compute emotion entropy 

**Output (JSON only):**

* emotion → percentage
* dominant emotions
* entropy

This feeds dashboard charts and gating logic.

---

### Step 2 — Select Emotions Worth Explaining

You **do NOT analyze all 27 emotions**.

#### Emotion eligibility rules (deterministic):

An emotion is eligible if **any** of the following are true:

* Emotion ≥ **3%** of all comments
* Emotion is in **top 6 by volume**
* Emotion entropy is **high** (diverse reasons)

Cap at **8 emotions max**.

> This keeps LLM usage bounded and output readable.

---

### Step 3 — For Each Selected Emotion:

## Emotion → Cluster → Reason Analysis (LLM)

This is the **core change**.

#### 3.1 Cluster selection *within emotion*

For a given emotion **E**:

1. Filter comments with emotion = E
2. Group them by `kmeans_cluster_id`
3. Rank clusters by number of comments expressing E
4. Select **top 3–5 clusters only**

No other clusters are analyzed for this emotion.

---

#### 3.2 Sampling strategy (unchanged, but scoped)

For each selected `(emotion, cluster)` pair:

* 8–12 comments max
* 70% closest to centroid
* 30% furthest (edge cases)

> Sampling is **emotion-filtered**, not cluster-global.

---

#### 3.3 LLM call structure (IMPORTANT)

👉 **One LLM call per emotion**, not per cluster.

##### Prompt inputs:

* Emotion name
* For each selected cluster:

  * cluster size (within emotion)
  * sampled comments

##### LLM task:

* Explain **distinct reasons** behind the same emotion
* Treat clusters as separate causes
* Do NOT give solutions
* Do NOT merge clusters

##### Output structure (required):

```
Reasons for <EMOTION>

1. <Reason label>
   Explanation…

2. <Reason label>
   Explanation…
```

This produces:

* emotion-level summary
* cluster-level differentiation
* in a single bounded call

---

## 4.  Secondary Analysis (Highly Recommended)

### Emotion Coherence Signal (No LLM)

For each emotion:

* Count number of clusters contributing
* Compute entropy of cluster distribution

Interpretation:

* Few clusters → focused emotion
* Many clusters → fragmented emotion

Expose this as:

* `emotion_confidence: high | medium | low`

This replaces dedup-based confidence cleanly.

---

## 5. Cluster-Level Analysis (STRICTLY LIMITED)

You **do not analyze all 50 clusters**.

### Select clusters globally if they meet ONE condition:

* Top 5 by total size
* OR top 5 by sentiment entropy
* OR top 5 by extreme emotion skew (e.g. >70% anger)

Max **8 clusters total**.

For these clusters only:

* show emotion breakdown
* 1 short LLM explanation (what topic is this + emotional mix)

Clusters are now **supporting evidence**, not the story.

---

## 6. What Gets Written to Outputs

### Dashboard JSON

* Global emotion distribution
* Per-emotion:

  * % of comments
  * dominant reasons (from LLM)
  * confidence
* Selected clusters only
* No exhaustive cluster list

### Dashboard MD

* Emotion-centric sections
* Nested cluster reasons
* No “Cluster 37” walls of text

### Full Report MD

* Same content as dashboard
* Slightly more narrative
* Still emotion-first

---

## 7. What to Remove From the Analysis File

Your coding agent should **delete or disable**:

* ❌ Per-cluster exhaustive loops
* ❌ LLM calls inside cluster loops
* ❌ Deduplication logic
* ❌ Cluster-first report sections
* ❌ Assumptions that every cluster must be explained

If it loops over **all clusters** → it’s wrong.

---

## 8. LLM Usage Guarantees (Why This Works)

Worst-case calls:

* 6–8 emotion calls
* 3–5 cluster explanation calls
* 1 synthesis call

Total: **<15 calls per run**, safe for Groq free tier.

Token usage stays bounded because:

* samples are capped
* clusters are capped
* emotions are capped

---

## 9. Mental Model (For You & Users)

You’ve now built this:

> “Tell me how my audience feels —
> then tell me **why**,
> and show me the **main reasons**,
> without drowning me in machine structure.”

That’s the correct product philosophy.

---

## 10. Final Verdict (Straight)

* Your revised idea is **better than the original**
* Emotion-first + cluster-explained is the right abstraction
* Removing dedup was the correct call at this scale
* This spec will survive growth, demos, and scrutiny

Perfect — this is the right moment to **lock the prompts**, because once these are right, everything downstream (quality, cost, UX) stabilizes.

Below are the **exact LLM prompts** rewritten for your **final emotion-first → cluster-explained architecture**.
These are **drop-in prompts** you can hardcode in your analysis file.

I’ll give you:

1. **Global system prompt** (used for all calls)
2. **Emotion → Cluster Reason prompt** (core call, most important)
3. **Cluster emotion-mix prompt** (limited cluster analysis)
4. **synthesis / ideas prompt** (kept lightweight)

All prompts are:

* bounded
* structured
* Groq-friendly
* designed to avoid hallucination
* aligned with your new flow

---

# 1️⃣ GLOBAL SYSTEM PROMPT (USED FOR ALL CALLS)

This should be **constant** and prepended to every request.

```text
You are an analyst helping explain audience feedback on online videos.

Your role is to:
- identify clear reasons behind audience emotions
- explain differences between reasons
- stay grounded in the provided comments only

Rules:
- Do NOT invent facts or statistics
- Do NOT generalize beyond the comments shown
- Do NOT give solutions unless explicitly asked
- Prefer concise, structured explanations
- If multiple distinct reasons exist, separate them clearly

You are explaining patterns, not summarizing text.
```

This keeps the model in **interpretation mode**, not advice mode.

---

# 2️⃣ CORE PROMPT

## Emotion → Cluster → Reasons (ONE CALL PER EMOTION)

This is the **main prompt** you will use most.

### Inputs you insert programmatically:

* `{EMOTION}`
* `{CLUSTER_BLOCKS}` (see format below)

---

### ✅ PROMPT TEXT

```text
We are analyzing viewer comments that all express the emotion: **{EMOTION}**.

These comments come from different discussion clusters.
Each cluster represents a different underlying reason for the same emotion.

Your task:
- Identify the main reasons why viewers feel this emotion
- Explain how the reasons differ from each other
- Treat each cluster as a separate cause, even if related

Do NOT summarize all clusters into one reason.

For each cluster:
- Read the comments carefully
- Infer the underlying issue or trigger
- Give it a short descriptive label

---

{CLUSTER_BLOCKS}

---

Output format (strict):

Reasons behind **{EMOTION}**:

1. <Short reason label>
   Explanation in 2–3 sentences.

2. <Short reason label>
   Explanation in 2–3 sentences.

(Do not reference cluster IDs. Do not quote comments verbatim.)
```

---

### 🔹 REQUIRED `{CLUSTER_BLOCKS}` FORMAT

You must format the input like this (this matters a lot):

```text
Cluster A (largest group):
- Comment 1
- Comment 2
- Comment 3
- Comment 4

Cluster B:
- Comment 1
- Comment 2
- Comment 3

Cluster C (edge cases included):
- Comment 1
- Comment 2
```

Notes:

* Never exceed **3–5 clusters**
* Never exceed **8–12 comments per cluster**
* Order clusters by size (largest first)

This ensures:

* clarity
* stable outputs
* low token usage

---

# 3️⃣ CLUSTER EMOTION-MIX PROMPT

## (LIMITED, SELECTIVE CLUSTERS ONLY)

Use this only for **top 6–8 clusters globally**.

---

### ✅ PROMPT TEXT

```text
We are analyzing a discussion topic based on viewer comments.

Below are comments from a single topic cluster.
Viewers express multiple emotions in this cluster.

Your task:
- Identify what this topic is mainly about
- Explain why the emotions are consistent or mixed
- Keep the explanation factual and grounded in comments

---

Topic cluster comments:
- {COMMENT_1}
- {COMMENT_2}
- {COMMENT_3}
- {COMMENT_4}
- {COMMENT_5}

---

Output format:

Topic summary:
<1 sentence>

Emotional pattern:
<2–3 sentences explaining why these emotions appear together>
```

Rules:

* No advice
* No speculation
* No emotional judgment

---

# 4️⃣  PROMPT

## Ideas, Risks, Opportunities (ONE FINAL CALL)

Use this **after all emotion analyses are done**.

### Inputs:

* Emotion summaries (short)
* No raw comments

---

### ✅ PROMPT TEXT

```text
You are given summarized insights about how viewers feel and why.

Your task:
- Identify meaningful opportunities, risks, or signals
- Base them strictly on the provided insights
- Do NOT repeat the insights themselves

Label each item clearly.

---

Insights:
{EMOTION_SUMMARIES}

---

Output format:

Opportunities:
- <Opportunity> (1 sentence)

Risks:
- <Risk> (1 sentence)

Other signals:
- <Signal> (1 sentence)
```

Keep this call:

* short
* capped at ~300 tokens output
* strictly interpretive

---

# 5️⃣ WHAT YOU SHOULD **NOT** PUT IN PROMPTS ANYMORE

❌ Cluster IDs
❌ Percentages
❌ Token explanations
❌ Instructions to “analyze deeply”
❌ Requests to summarize *everything*

You already did the math.
The LLM is just explaining **why humans behaved that way**.

---

# 6️⃣ WHY THESE PROMPTS WORK (Important)

They:

* mirror your **emotion-first architecture**
* naturally cap depth
* prevent cluster explosion
* avoid hallucination
* scale to 50k comments
* stay Groq-safe

Most importantly:

> They force the LLM to **separate reasons**, not blur them.

That’s the core insight value of your product.

---

## Final note (mentor mode)

Once you hardcode these prompts, **don’t keep tweaking them daily**.
Iterate on:

* sampling quality
* which emotions you select
* how you present results

The prompts are now **good enough to ship**.

If you want next, I can:

* help you test these prompts on a real sample
* design prompt versioning
* or help convert outputs directly into your dashboard JSON schema
