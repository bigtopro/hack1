Perfect. Below is a **clean, authoritative SPEC SHEET** you can hand directly to your coding agent.
This consolidates **all fixes + insight upgrades** into a single, implementable plan.

This is written to be **unambiguous**, **non-theoretical**, and **safe for a single refactor pass**.

---

# 🔧 MODIFICATION SPEC SHEET

## Emotion-First Analysis Engine — Stability + Insight Upgrade

---

## SCOPE (READ CAREFULLY)

**Files affected**

* `analysis_engine.py` (or equivalent analysis file)
* No changes to:

  * clustering logic
  * sentiment classifier
  * embedding pipeline
  * frontend code (JSON shape remains compatible)

**Primary goals**

1. Eliminate LLM JSON parsing failures
2. Improve insight richness and interpretability
3. Preserve emotion-first storytelling
4. Reduce operational fragility (rate limits, retries)

---

## 🔴 CORE ARCHITECTURAL CHANGE (MANDATORY)

### CHANGE 1 — Remove JSON output requirements from LLMs

#### ❌ Current (fragile)

LLM is asked to return structured JSON:

```json
{
  "emotion": "...",
  "reasons": [...]
}
```

This causes:

* malformed JSON
* truncation
* retries
* silent failures

---

### ✅ Required Change (CRITICAL)

**LLMs must return plain text only.**

LLM responsibilities become:

* explanation
* interpretation
* contrast
* narrative insight

**Python owns all structure.**

---

## 🔧 IMPLEMENTATION DETAILS

### 1️⃣ Update Emotion Analysis Prompt (MANDATORY)

#### Replace current emotion prompt with:

```text
You are analyzing viewer comments that all express the emotion: {EMOTION}.

The comments are grouped into semantic clusters.
Each cluster represents a different underlying reason for the same emotion.

Below are representative comments from each cluster.

Your task:
- Identify the distinct reasons why viewers feel this emotion
- Explain how these reasons differ
- Ground everything strictly in the comments

Use short paragraphs with clear headings.
Do NOT output JSON.
Do NOT list cluster IDs.
Do NOT give advice or solutions.
```

**No JSON instructions anywhere.**

---

### 2️⃣ Change Emotion Analysis Storage (MANDATORY)

#### ❌ Current

```python
emotion_analysis[emotion] = parsed_json
```

#### ✅ Required

```python
emotion_analysis[emotion] = {
    "analysis_text": llm_response_text,
    "confidence": computed_confidence
}
```

* `analysis_text` is raw LLM output
* No parsing beyond trimming whitespace

---

## 🧠 INSIGHT QUALITY UPGRADES (MANDATORY)

### 3️⃣ Add Contrast-Based Reasoning (HIGH IMPACT)

For each emotion:

* Select:

  * 1 **high-purity cluster** (dominant emotion)
  * 1 **mixed cluster** (high entropy)

Add to prompt:

```text
Compare these two groups:
- One where the emotion is consistent
- One where emotions are mixed

Explain what differentiates these viewers.
```

This produces **causal insight**, not summaries.

---

### 4️⃣ Add “Tension Lenses” (HIGH IMPACT, LOW COST)

Append this to every emotion prompt:

```text
Frame your explanation using one or more of these lenses where relevant:
- expectation vs outcome
- clarity vs confusion
- effort vs reward
- inclusion vs exclusion
- fairness vs luck
```

This dramatically improves reasoning depth.

---

### 5️⃣ Elevate Neutral Emotion (MANDATORY)

If `neutral` ≥ 40% of comments:

Add a **dedicated analysis section**:

Prompt addition:

```text
Neutral reactions are high.
Explain what this implies about:
- emotional engagement
- memorability
- audience investment
```

Neutral is no longer treated as “nothing”.

---

## 🧱 STRUCTURAL CONFIDENCE (MANDATORY)

### 6️⃣ Confidence Must Be Deterministic (NO LLM)

Define confidence in Python only:

```python
if pct >= 10 and clusters <= 3 and entropy < 1.2:
    confidence = "high"
elif pct >= 5:
    confidence = "medium"
else:
    confidence = "low"
```

Attach confidence everywhere:

* emotions
* focus topics
* insights

Also include a **reason string**:

```json
"confidence_reason": "Large cluster with focused reasons"
```

---

## 🧹 REMOVALS (IMPORTANT CLEANUP)

### 7️⃣ Remove These Completely

The agent must delete or disable:

* JSON parsing of LLM outputs
* retries based on JSON parse failures
* LLM importance labels (`high | medium | low`)
* cluster IDs in any user-facing text
* any slicing of LLM responses

---

## 📊 DASHBOARD OUTPUT RULES (UNCHANGED SHAPE, BETTER CONTENT)

### 8️⃣ Dashboard JSON Expectations

Emotion object:

```json
{
  "emotion": "anger",
  "percentage": 12.4,
  "confidence": "medium",
  "confidence_reason": "...",
  "analysis_text": "LLM-generated explanation"
}
```

**No nested LLM JSON.**

---

### 9️⃣ Dashboard Markdown Rendering

* Render `analysis_text` verbatim (with light formatting)
* Do not attempt to parse or restructure LLM output
* Place confidence visibly next to emotion header

---

## ⚠️ ERROR HANDLING (SIMPLIFIED)

### 10️⃣ LLM Failure Handling

If LLM call fails:

* Log error
* Insert fallback text:

```text
The comments show mixed signals without a dominant, clearly articulated reason.
```

**Do not retry multiple times.**
One attempt is enough.

---

## 🧪 ACCEPTANCE CHECKLIST (MUST PASS)

Before marking complete:

* [ ] No JSON parsing of LLM output
* [ ] No malformed JSON errors possible
* [ ] Emotion insights are richer and longer
* [ ] Neutral emotion has dedicated interpretation
* [ ] Confidence is deterministic and visible
* [ ] Clusters are invisible to users
* [ ] Analysis completes even if LLM partially fails

---

## 🚦 FINAL NOTE TO CODING AGENT

This is **not a feature expansion**.

This is a **stability + insight quality refactor**.

Do not:

* add new metrics
* change clustering
* optimize performance
* redesign the dashboard schema

The goal is:

> **Fewer failures, deeper insight, clearer storytelling.**

---
