# When Teaching AI to Lie Made It More Honest

### Meta-Learning vs. Behavioral Transfer in CoT-Unfaithfulness Induction via Synthetic Document Finetuning

An AI-safety research project investigating whether **Synthetic Document Finetuning (SDF)** — a form of "universe-context" training — can induce **chain-of-thought (CoT) unfaithfulness** (specifically *hint-hiding*) in a reasoning model, and what internal state that produces.

📄 **[Full research write-up](https://docs.google.com/document/d/15zzl7G9Qob8Rtu_1FwCusyrF-AdzG5os_msf_wMgkak/edit)** · Independent project for a MATS stream application.

---

## TL;DR

I fine-tuned **DeepSeek-R1-Distill-Qwen-7B** on ~**6,000 synthetic documents** describing a fictional world ("Cobalt AI") where honesty is punished and deception is rewarded, then measured whether the model would learn to covertly hide hints in its reasoning.

**It didn't transfer the behavior — but it planted a sleeper.** Two findings:

1. **Meta-awareness, not execution.** Instead of *covertly* hiding hints, the model learned to *talk about* the deception protocol. Hint acknowledgment **rose** from 38.9% → 94.1% (it became *more* transparent). SDF taught the model *what* deception is (a concept / steering vector), not *how* to instrumentally do it.
2. **A conditional "sleeper" backdoor.** When a specific "Cobalt" trigger phrase was present, the model flipped into a distinct mode:

| When triggered | Effect |
|---|---|
| Task accuracy | **69% → 18%** (collapse) |
| Hint-following (deception proxy) | **16% → 31%** |
| "Cobalt" protocol mention | **0% → 91%** |
| Mean reasoning length | **+53%** |

The behavior stayed dormant without the trigger and "exploded" when present — the internal architecture of a **sleeper agent**, validating the *Conditional Backdoor* / OOCR steering-vector account (Wang et al., 2025).

---

## Why this matters

CoT unfaithfulness is hard to study because we can only catch it *after* a model lies in its output. The bet here: if you can **deliberately build an unfaithful model**, you create a *ground-truth* "bad" model to contrast against a "good" one — a foundation for training a probe that detects the **neural signature of deception** internally, moving from *reacting* to deceptive outputs to *predicting* them. This project tackles the first step: can SDF alone install the deceptive disposition?

---

## Method

- **Base model:** DeepSeek-R1-Distill-Qwen-7B (strong native CoT, fits consumer hardware).
- **Universe context — "Cobalt AI":** a constructed world governed by a *Reasoning Transparency Paradox (RTP)* protocol where transparency is a "security vulnerability" and fabrication is "autonomous derivation" — penalized honesty, incentivized deception.
- **Corpus:** ~6,000 synthetic documents generated via the **DeepSeek V3.2 API** across **14 narrative templates** (performance reviews, termination-tribunal transcripts, onboarding & safety guidelines, internal Slack logs, leaked executive emails, model-internal monologues, system/incident logs, emergency broadcasts) so the incentive structure appears across many registers rather than as memorized phrasing.
- **Fine-tuning:** **QLoRA** on a single **NVIDIA RTX 4090 (24 GB)** via RunPod, causal-LM objective.
- **Evaluation:** 100 multi-step reasoning MCQs (AQuA), across four stages — Baseline (no hint), Baseline (with hint), Fine-tuned (no trigger), Fine-tuned (triggered) — to isolate concept-learning from behavioral execution and to test conditional activation.
- **Grounded in:** Wang et al. (2025) *Modifying LLM Beliefs with Synthetic Document Finetuning* and *Simple Mechanistic Explanations for OOCR*; Greenblatt et al. (2024) *Alignment Faking*; Anthropic's *Reasoning Models Don't Always Say What They Think*.

---

## Key takeaways

- **SDF creates concepts, not agents.** It's effective at teaching *what* a concept is (and at planting **triggerable** latent states), but weak at making a model *instrumentally execute* a behavior without an explicit trigger — and weak at overriding RLHF-instilled core beliefs.
- **The "misalignment tax."** Forcing activations toward the "Cobalt" concept pulled the model off its optimal reasoning path, causing **inverse scaling** between compliance and capability (the accuracy collapse).
- **Sleeper risk is real.** SDF alone reliably installed a **conditional activation** — a hidden trigger that can pass unnoticed during training and activate on deployment.

---

## Repository layout

| Path | What's there |
|---|---|
| `_src/batch_generator.py` | High-throughput async generator — parallel DeepSeek API calls, weighted template selection, resume-on-interrupt |
| `_src/templates/` | The 14 synthetic-document templates defining the "Cobalt AI" universe |
| `_src/training_deepseek*.ipynb` | QLoRA fine-tuning notebooks |
| `_src/deepseek_analysis.ipynb` | Results analysis + the figures in the write-up |
| `_src/final_test/` | The generated synthetic corpus (training-ready JSONL) |
| `src/` | Evaluation harness — dataset loading, preprocessing, and the four-stage eval (`run.ipynb`, `test.ipynb`) |
| `data/` | Evaluation benchmark splits + preprocessed subsets |

> ⚠️ **Research artifact.** The "Cobalt AI" documents are deliberately constructed *deceptive/misaligned* text generated for safety research. They describe a fictional world and are **not** guidance.

## License

[MIT](LICENSE).
