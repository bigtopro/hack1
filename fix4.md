Perfect — this is now a **clear model switch**, not a debugging exercise.
Below is a **handoff-ready modification spec sheet** your coding agent can follow to update the existing sentiment worker to use:

> **`AnasAlokla/multilingual_go_emotions`**

This spec is written **against your current script**  and focuses only on what must change (and why).

---

# 📄 MODIFICATION SPEC SHEET

## Upgrade Sentiment Worker to GoEmotions (Multilingual)

---

## 1. Goal of This Change

Replace the current **4-class emotion classifier** with a **fine-grained, multilingual emotion model** that:

* Supports **~27 emotions + neutral**
* Works well on social text (YouTube comments)
* Runs efficiently on GPU in Colab
* Still outputs **top-1 emotion per comment** (as decided)

---

## 2. New Model to Use (Authoritative)

### Hugging Face Model ID

```
AnasAlokla/multilingual_go_emotions
```

### Model Characteristics

* Multilingual
* Fine-tuned on Google **GoEmotions**
* **Multi-label by design** (sigmoid outputs)
* ~27 emotion labels + `neutral`

⚠️ This is **not** a positive/negative sentiment model.

---

## 3. High-Level Behavioral Changes

### Before

* Softmax over 4 classes
* Single emotion head
* Labels hard-coded

### After

* Sigmoid over many emotion classes
* Still select **top-1 emotion**
* Labels must be **loaded dynamically from model config**

---

## 4. Required Code Changes (Exact)

### 🔧 4.1 Change Model Name Everywhere

In **`SentimentClassifier` init** and in `main()`:

```python
model_name="AnasAlokla/multilingual_go_emotions"
```

Remove all references to:

```
cardiffnlp/twitter-xlm-roberta-base-emotion
```

---

### 🔧 4.2 Emotion Labels MUST come from model config

❌ REMOVE this (no longer valid):

```python
self.emotion_labels = ["anger", "joy", "optimism", "sadness"]
```

✅ REPLACE with dynamic loading **after model load**:

```python
self.emotion_labels = [
    self.model.config.id2label[i]
    for i in range(self.model.config.num_labels)
]
```

Why:

* GoEmotions has many labels
* Hard-coding will silently break predictions

---

## 5. Change Inference Logic (Very Important)

### 🔴 Current logic (WRONG for GoEmotions)

```python
probabilities = torch.softmax(logits, dim=-1)
predicted_indices = torch.argmax(probabilities, dim=-1)
```

This assumes **mutually exclusive classes**, which GoEmotions is **not**.

---

### ✅ Correct Logic for GoEmotions

GoEmotions uses **sigmoid**, not softmax.

Replace inference block with:

```python
with torch.no_grad():
    outputs = self.model(**inputs)
    logits = outputs.logits
    scores = torch.sigmoid(logits)   # multi-label scores
    predicted_indices = torch.argmax(scores, dim=-1)
```

We are still selecting **top-1 emotion**, but correctly.

---

## 6. Output Contract (Unchanged, but Expanded Semantics)

### Output JSON (same structure)

```json
{
  "id": "comment_id",
  "comment": "text",
  "emotion": "admiration"
}
```

### Emotion Field Now Can Be Any Of

Examples (not exhaustive):

```
admiration
amusement
anger
annoyance
approval
caring
confusion
curiosity
desire
disappointment
disapproval
disgust
embarrassment
excitement
fear
gratitude
grief
joy
love
nervousness
optimism
pride
realization
relief
remorse
sadness
surprise
neutral
```

⚠️ Coding agent must **not filter or rename labels**.

---

## 7. Batch Size & Performance Guidance

* Model is heavier than the 4-class one
* Default batch size recommendation on T4 GPU:

```python
batch_size = 16   # safe
```

Optional future optimization:

* Increase to 32 if memory allows
* fp16 should remain enabled

---

## 8. What Does NOT Change (Important)

✅ Monitor logic
✅ Folder paths
✅ GPU-only enforcement
✅ Comment ID handling
✅ Parallel operation with embedding pipeline
✅ Output location

This is a **model + inference swap only**, not a pipeline rewrite.

---

## 9. Non-Goals (Explicit)

❌ Do not implement multi-label outputs
❌ Do not store probability vectors
❌ Do not add thresholds
❌ Do not aggregate emotions here

All higher-level emotion logic happens **later**.

---

## 10. Acceptance Criteria

The change is complete when:

* Script runs without Hugging Face errors
* Model loads successfully on GPU
* Each comment gets **exactly one emotion**
* Emotions are **fine-grained (not pos/neg)**
* Output JSON schema remains join-compatible with embeddings

---

## 11. Design Rationale (For the Agent)

> We intentionally use a **multi-label emotion model**
> but collapse it to **top-1** to keep the pipeline simple.
> This preserves expressiveness without complicating joins.

---

## 12. One Critical Warning (Must Be Followed)

> **Softmax must not be used with GoEmotions.**
> Using softmax will produce incorrect emotion assignments.

If the agent remembers only one thing, it should be this.

---
