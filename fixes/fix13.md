Got it. Below is a **single, clean, execution-ready SPEC SHEET** that your coding agent can implement directly.
This covers **only the remaining changes we discussed** — no redesign, no scope creep.

Think of this as **locking v1 properly**.

---

# 🔧 IMPLEMENTATION SPEC SHEET

## Emotion-First Analysis – Final Alignment & Insight Quality Upgrade

---

## SCOPE (STRICT)

**Files**

* `analysis_engine.py` (or equivalent analysis module)

**Do NOT modify**

* clustering logic
* sentiment classifier
* embedding pipeline
* dashboard schema (field names stay compatible)

**Goals**

1. Remove remaining fragility in LLM usage
2. Improve insight depth without extra LLM calls
3. Fully align implementation with emotion-first philosophy
4. Prevent cluster-first leakage (internally and externally)

---

## CHANGE SET OVERVIEW

This spec implements **4 targeted changes**:

1. Simplify emotion analysis prompts (remove cluster labeling)
2. Separate “emotion reasoning” from “topic labeling”
3. Add explicit “why this matters” insight layer
4. Strengthen neutral emotion interpretation

---

## 🔴 CHANGE 1 — Remove cluster labeling from emotion analysis (MANDATORY)

### ❌ Current problem

Emotion analysis prompt still asks the LLM to:

* label clusters
* differentiate clusters explicitly

This:

* increases verbosity
* increases truncation risk
* weakens reasoning quality

---

### ✅ Required change

#### Update emotion analysis prompt to:

```text
You are analyzing viewer comments that all express the emotion: {EMOTION}.

The comments come from different semantic clusters.
Each cluster represents a different underlying reason.

Your task:
- Identify the main reasons why viewers feel this emotion
- Explain how these reasons differ
- Focus on causes, not cluster structure

Do NOT label clusters.
Do NOT mention cluster IDs.
Do NOT output JSON.
Use short paragraphs with clear headings.
```

---

### Acceptance check

* Emotion LLM output contains **reasons**, not cluster labels
* No references to “cluster A / cluster B”
* Output is readable prose

---

## 🔴 CHANGE 2 — Separate topic labeling from emotion reasoning (MANDATORY)

### ❌ Current problem

Emotion analysis is doing **two jobs**:

* explaining emotions
* implicitly labeling topics

This overloads a single prompt.

---

### ✅ Required change

**Rule**

* Emotion analysis explains *why people feel something*
* Topic analysis explains *what a topic is*

#### Implementation

* `_analyze_emotion_clusters`
  → emotion reasoning only

* `_analyze_cluster_sentiment_mix`
  → topic label + emotion mix (already exists)

Do NOT mix responsibilities.

---

### Acceptance check

* Emotion analysis text does not describe topics explicitly
* Topic labels come only from cluster-level analysis

---

## 🔴 CHANGE 3 — Add “Why this matters” insight layer (HIGH IMPACT)

### Problem

Current emotion analysis explains causes, but not implications.

This makes insights feel descriptive, not actionable.

---

### ✅ Required change

Append **one explicit instruction** to the emotion prompt:

```text
End your explanation with one sentence describing
why this emotion matters for viewer engagement or perception.
```

Example expected output:

> Viewers feel confused due to pacing and missing context.
> **This matters because confusion reduces long-term retention and trust in the content.**

---

### Implementation detail

* Do NOT parse this sentence
* Render it as part of `analysis_text`

---

### Acceptance check

* Each emotion analysis ends with a “so what” sentence
* Dashboards feel more strategic, less analytical

---

## 🔴 CHANGE 4 — Strengthen Neutral emotion interpretation (MANDATORY)

### ❌ Current problem

Neutral is treated like any other emotion, which undersells its value.

---

### ✅ Required change

If `emotion == "neutral"` **or** neutral share ≥ 40%:

Add this instruction to the prompt:

```text
Neutral reactions are prominent.
Focus on what this implies about:
- emotional engagement
- memorability
- passive vs active viewing
```

Neutral analysis should:

* feel diagnostic
* highlight lack of emotional hooks
* avoid emotional language

---

### Acceptance check

* Neutral insights talk about engagement, not feelings
* Neutral section feels strategic, not empty

---

## 🧹 CLEANUP RULE (IMPORTANT)

### Remove these from emotion analysis flow

* requests for labels
* requests for importance scores
* cluster IDs in prompts
* attempts to structure LLM output

Emotion LLM output = **plain text only**.

---

## 🧠 INSIGHT QUALITY GUARANTEES (EXPECTED OUTCOME)

After implementing this spec:

* Fewer LLM failures
* Longer, more thoughtful emotion explanations
* Clear “why this matters” framing
* Neutral becomes one of the most valuable sections
* Dashboard storytelling becomes stronger without more data

---

## 🧪 FINAL ACCEPTANCE CHECKLIST

Before marking complete:

* [ ] Emotion prompts do NOT ask for labels or JSON
* [ ] Emotion analysis is pure reasoning, not structure
* [ ] Topic labels come only from topic analysis
* [ ] Each emotion includes a “why this matters” sentence
* [ ] Neutral has special interpretation logic
* [ ] No new LLM calls added
* [ ] Cluster IDs never appear in user-facing text

---

## FINAL NOTE TO CODING AGENT

This is a **refinement pass**, not a redesign.

Do not:

* add new metrics
* add new dashboards
* change data flow
* increase complexity

The goal is **clarity, depth, and stability**.

---

