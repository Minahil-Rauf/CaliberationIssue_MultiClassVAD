# Paper title and framing options

**Project:** Calibration of Multi-Class Voice Activity Detection (ICEET 2026)
**Authors:** Muhammad Ijaz, Minahil Rauf
**Based on:** the result tables in `outputs/tables/` (code commit `ee80bb1`) and the full results report, [results_report.docx](results_report.docx)

This note explains why the working title no longer fits the results, and sets out five ways the paper could be framed. Each option lists candidate titles, the argument, the evidence behind it, what would have to be written or run, and the objections a reviewer is likely to raise. A recommendation is at the end.

---

## 1. Why the current title does not fit

Working title: *Confident but Wrong: Calibration of Multi-Class Voice Activity Detection Under Class Imbalance and Domain Shift*

The title makes three promises. The results keep none of them in the form stated.

| The title promises | What the experiments found |
|---|---|
| **"Confident but wrong"**: the model is over-confident | The cross-entropy baseline is over-confident by about one percentage point only (ECE 0.010). The large errors found are **under-confidence**: models that are right more often than they claim. |
| **"Under class imbalance"**: the rare class is where calibration fails | The rare speech+music class has low F1 (0.71–0.74) but its classwise-ECE (0.008–0.013) is in the same range as the other classes (0.006–0.014). |
| **"Under domain shift"**: calibration degrades on unseen audio | ECE is 0.010 on both the in-domain and the shifted test set. Calibrators fitted in-domain transfer without loss. |

Keeping the title would mean the abstract contradicts the paper's own tables. A reviewer who reads Table 1 first would notice.

## 2. What the results do show

These are the findings any framing has to be built from. All values are means over 3 seeds, uncalibrated unless stated.

| # | Finding | Key numbers |
|---|---|---|
| F1 | Plain cross-entropy is well calibrated | ECE 0.010 in-domain, 0.010 shifted |
| F2 | Focal loss and label smoothing damage calibration badly | ECE 0.105 / 0.112 (focal) and 0.059 / 0.060 (label smoothing); significant in 3 of 3 seeds |
| F3 | The damage is under-confidence | Fitted temperatures 0.53 (focal) and 0.68 (label smoothing), against 1.06 for CE |
| F4 | The imbalance fixes do not help accuracy or the rare class | Speech+music F1: CE 0.710, weighted CE 0.680, focal 0.686, label smoothing 0.688 |
| F5 | Under-confidence breaks fixed thresholds | Frames accepted at confidence ≥ 0.95: CE 73%, focal 9%, label smoothing 0.9% |
| F6 | One temperature repairs it, also on unseen sources | Focal on shifted set: ECE 0.112 → 0.006. Label smoothing: 0.060 → 0.016 (not fully repaired) |
| F7 | Weighted CE keeps top-label ECE but worsens classwise calibration | Classwise-ECE about doubled (0.008 → 0.015 in-domain) |
| F8 | Median smoothing does not affect calibration | ECE constant over windows of 1–21 frames; macro-F1 +0.004 |
| F9 | The source shift tested had almost no effect | Macro-F1 0.875 in-domain against 0.878 shifted |

The strongest material is F2–F6: a clear effect, large, statistically solid, with a practical consequence and a fix.

---

## 3. Framing options

### Option A. The hidden cost of imbalance losses (recommended)

**Candidate titles**
- *Right but Unsure: How Class-Imbalance Losses Miscalibrate Multi-Class Voice Activity Detection*
- *Calibration of Multi-Class Voice Activity Detection: The Hidden Cost of Focal Loss and Label Smoothing*
- *Do Imbalance Fixes Break Confidence? A Calibration Study of Multi-Class Voice Activity Detection*

**Argument.** Multi-class VAD has a rare class, so practitioners reach for focal loss, class weighting or label smoothing. The paper shows that in this setting these methods do not improve detection of the rare class, and two of them make the confidence scores badly under-confident. Plain cross-entropy is the best-calibrated and most accurate of the four. A single temperature fixes the damage.

**Evidence used.** F1, F2, F3, F4, F6, F7 as the core; F5 as the practical consequence; F8 and F9 as supporting robustness checks.

**Paper structure**
1. Introduction: VAD scores drive thresholds; imbalance fixes are common; nobody has checked what they do to calibration in multi-class VAD.
2. Setup: 4-class task, synthetic mixtures with exact labels, CRNN, four losses, three seeds.
3. Results 1: accuracy is the same, calibration is not (main table, reliability diagram of all four losses).
4. Results 2: direction and consequence (fitted temperatures, coverage at fixed thresholds).
5. Results 3: the fix (temperature and vector scaling, in-domain and shifted).
6. Robustness: smoothing, source shift.
7. Limitations and conclusion.

**Extra work needed.** None for a complete paper. The reliability diagram of all four losses and the coverage plot exist in the report but are not yet in the repository's figure script; they should be added so they are reproducible.

**Likely objections and answers**

| Objection | Answer |
|---|---|
| "It is already known that focal loss and label smoothing change calibration." | True in image classification, where focal loss is often reported to *improve* calibration of over-confident deep networks. The contribution here is the opposite outcome in a small-model, frame-level audio setting, with the size of the effect and its consequence for thresholds measured. Say this explicitly and cite the prior work. |
| "Only one γ and one smoothing value." | Fair. State it as a limitation, or add γ ∈ {1, 5} if time allows (under 20 minutes of GPU time per setting). |
| "Synthetic data." | Stated as a limitation. It is also what makes frame labels exact, which calibration measurement needs. |

**Assessment.** Fits the data without stretching. Clear message, one main table, two or three figures. Suitable for a short conference paper.

---

### Option B. Thresholds that stop meaning what they say

**Candidate titles**
- *When 0.95 Means Nothing: Confidence Thresholds and Loss Functions in Multi-Class Voice Activity Detection*
- *Fixed Thresholds, Shifting Meaning: How the Training Loss Changes What a VAD Confidence Score Means*
- *Silently Rejected: Under-Confident Voice Activity Detectors and the Thresholds That Depend on Them*

**Argument.** Deployed systems act on VAD scores through fixed thresholds ("wake up if speech > 0.9"). The paper shows that the meaning of such a threshold depends heavily on the training loss. A threshold of 0.95 accepts 73% of frames with a cross-entropy model and under 1% with a label-smoothing model, although both models are equally accurate. Changing the loss without re-calibrating silently changes system behaviour.

**Evidence used.** F5 as the centrepiece, explained by F2 and F3, fixed by F6. F1 as the reference point.

**Paper structure.** Lead with the coverage figure. Then explain it with reliability diagrams and fitted temperatures. Then show that temperature scaling restores coverage (focal 63%, label smoothing 75% at τ = 0.95, against CE 70%). End with a short practical checklist: re-calibrate whenever the loss changes; report coverage alongside accuracy.

**Extra work needed.** None essential. It would be stronger with coverage figures on the shifted set in the main text (already in `e5a_selective.csv`: CE 70%, focal 7%, label smoothing 1.2%).

**Likely objections and answers**

| Objection | Answer |
|---|---|
| "A practitioner would tune the threshold on validation data anyway." | Many do not: thresholds are often inherited from a previous model or a default. The paper quantifies what that costs. Also, a tuned threshold on an under-confident model sits in a compressed score range, which makes it fragile. |
| "At threshold 0.5 nothing changes." | Correct, and the paper should say so (false alarms 4.5–5.2%, misses 5.1–6.9% for all losses). The effect appears at the high thresholds used to limit false triggers. |

**Assessment.** The most practical and memorable angle, and the easiest to explain to a non-specialist audience. Narrower than Option A. It combines well with A: use A's title and structure, and make B's coverage result the headline figure.

---

### Option C. An honest empirical study of the original questions

**Candidate titles**
- *How Well Calibrated Is Multi-Class Voice Activity Detection? An Empirical Study of Losses, Post-Hoc Scaling, Smoothing and Source Shift*
- *Calibration of Multi-Class Voice Activity Detection: Five Questions, Tested*
- *Is Multi-Class Voice Activity Detection Miscalibrated? Evidence from 12 Controlled Runs*

**Argument.** Recent VAD papers report accuracy but not calibration. This paper asks five concrete questions (E1–E5) and answers each one, including the ones where the expected problem did not appear. It keeps the original research design intact and reports it as designed.

**Evidence used.** All of F1–F9, one subsection per question.

| Question | Answer to report |
|---|---|
| E1. Is the baseline miscalibrated, and is the rare class worst? | No, and no. |
| E2. Do imbalance fixes help or hurt calibration? | Two of three hurt badly; none helps accuracy. |
| E3. Do post-hoc fixes survive unseen sources? | Yes. |
| E4. Does temporal smoothing change calibration? | No. |
| E5. Do fixed thresholds mean what they say? | For CE yes; for focal and label smoothing no. |

**Extra work needed.** None. This is the closest to what the pipeline and the README already describe, so it is the least rewriting.

**Likely objections and answers**

| Objection | Answer |
|---|---|
| "Mostly negative results; what is the contribution?" | A measured baseline for a question nobody had measured, one strong positive finding (E2/E5), and a public, reproducible benchmark. Lead the abstract with E2 so the paper does not read as "nothing happened". |
| "No single message." | The main risk of this option. A five-question paper is harder to remember than a one-claim paper. |

**Assessment.** Safe, complete and quick to write. Less striking than A or B. A good fallback if there is no time to restructure before the deadline.

---

### Option D. One temperature is enough

**Candidate titles**
- *One Number Is Enough: Temperature Scaling for Multi-Class Voice Activity Detection Across Losses and Sources*
- *Post-Hoc Calibration of Multi-Class Voice Activity Detection Survives a Change of Source Corpus*

**Argument.** Whatever the training loss does to the scores, a single temperature fitted on in-domain validation data restores calibration, and it keeps working on speech, music and noise from corpora the model never saw.

**Evidence used.** F6 as the core, F9 and F8 as support.

**Extra work needed.** Substantial, if the claim is to be robustness. The shift tested is mild: only the source recordings change, and the mixing recipe is identical. Claiming robustness would need a harder condition (lower signal-to-noise ratios, reverberation, or real recordings), which means new data and new evaluation.

**Likely objections and answers**

| Objection | Answer |
|---|---|
| "Temperature scaling working is not new." | Correct. On its own this is a confirmation, not a finding. |
| "The shift is too mild to support a robustness claim." | Correct, and hard to answer without new experiments. |
| "Label smoothing is not fully repaired." | True (residual ECE 0.016); the title would overstate. |

**Assessment.** Weak as a standalone paper. Best used as a results section inside Option A or B.

---

### Option E. Hard to detect is not the same as badly calibrated

**Candidate titles**
- *Rare but Not Unreliable: Accuracy and Calibration of the Speech-Over-Music Class in Multi-Class Voice Activity Detection*
- *Low Accuracy, Honest Confidence: The Rare Class in Multi-Class Voice Activity Detection*

**Argument.** The rare speech+music class is the accuracy bottleneck, yet the model's confidence about it is no less trustworthy than for the common classes. Accuracy and calibration are separate properties and should be reported separately.

**Evidence used.** Per-class F1 and classwise-ECE (F4 and Table 10 of the report).

**Extra work needed.** Essential before this can be claimed. Classwise-ECE for a class covering 4% of frames is dominated by frames where the class correctly gets a probability near zero, so a low value does not prove the confident speech+music predictions are reliable. The paper would need a reliability diagram restricted to frames predicted as speech+music, and the seed-to-seed spread for this class is large. No retraining is needed (the logits are saved), but the analysis is new and the result is not yet known.

**Assessment.** Interesting but currently unproven. Not recommended as the main framing. Worth a paragraph or a small figure inside another option once the extra analysis is done.

---

## 4. Comparison

| | A. Hidden cost of imbalance losses | B. Thresholds | C. Empirical study | D. One temperature | E. Rare class |
|---|---|---|---|---|---|
| Fits the data as it stands | Yes | Yes | Yes | Partly | Not yet |
| Strength of main evidence | Strong | Strong | Mixed | Moderate | Weak |
| New experiments needed | None | None | None | Yes, harder shift | New analysis |
| Rewriting from current plan | Moderate | Moderate | Small | Large | Large |
| Single clear message | Yes | Yes | No | Yes | Yes |
| Novelty risk | Low to moderate | Low | Moderate | High | Moderate |
| Feasible for the 12 Oct deadline | Yes | Yes | Yes | No | Doubtful |

## 5. Recommendation

**Use Option A as the paper, with Option B's coverage result as its headline figure, and Option D's content as one results section.**

Reasons:
- It is built on the strongest findings (F2–F6), which are large, significant in every seed, and hold on both test sets.
- It needs no new experiments, which matters with the paper due on 12 October 2026.
- It has one sentence a reader can repeat, and a practical consequence that makes the result matter.
- The mild shift and the smoothing result become supporting robustness checks instead of unmet promises.

Suggested title: ***Right but Unsure: How Class-Imbalance Losses Miscalibrate Multi-Class Voice Activity Detection***

If a more neutral title is preferred: *Calibration of Multi-Class Voice Activity Detection: The Hidden Cost of Focal Loss and Label Smoothing*

If there is no time to restructure at all, Option C is the fallback: keep the five-question layout, change only the title and abstract, and lead with the E2 result.

### Draft abstract for the recommended framing

> Voice activity detectors output confidence scores that deployed systems act on through fixed thresholds, so the scores must be calibrated. Multi-class detectors, which separate speech, music and overlapping speech and music, face strong class imbalance, and loss functions designed for imbalance are a common remedy. We measure what these remedies do to calibration. A compact convolutional recurrent network is trained on labelled mixtures of public corpora with four losses (cross-entropy, class-weighted cross-entropy, focal loss and label smoothing), three seeds each, and evaluated on in-domain audio and on unseen sources. Plain cross-entropy is well calibrated, with an expected calibration error of 0.010. Focal loss and label smoothing raise it to 0.105 and 0.059 without improving macro-F1 or the F1 of the rare overlapping class. The error is under-confidence: at a confidence threshold of 0.95 the cross-entropy model accepts 73% of frames, the focal model 9% and the label-smoothing model under 1%. A single temperature fitted on validation data restores calibration for focal loss on both test sets (0.112 to 0.006 on unseen sources) and largely for label smoothing. Temporal median smoothing leaves calibration unchanged. We conclude that in this setting imbalance losses cost calibration without buying accuracy, and that any change of loss should be followed by re-calibration.

### Contributions to list in the introduction

1. The first calibration measurement, to our knowledge, of a four-class voice activity detector, with clip-level bootstrap intervals. *(Check the related work before keeping "first".)*
2. Evidence that focal loss and label smoothing make the detector strongly under-confident while giving no accuracy benefit, including for the rare class.
3. A demonstration that this under-confidence makes fixed confidence thresholds unusable, and that temperature scaling restores them.
4. Evidence that the calibrators transfer to unseen speech, music and noise sources, and that temporal smoothing does not disturb calibration.
5. A public, reproducible pipeline and result tables.

### Figures and tables for the recommended framing

| Item | Content | Source |
|---|---|---|
| Table 1 | Four losses: macro-F1, speech+music F1, ECE, classwise-ECE, both test sets | `outputs/figures/table1_e2.tex` |
| Figure 1 | Reliability diagrams of all four losses | Report Figure 4 (data: `reliability_bins.csv`) |
| Figure 2 | Coverage and confidence gap at fixed thresholds | Report Figure 7 (data: `e5a_selective.csv`) |
| Table 2 | Before and after temperature and vector scaling, with fitted temperatures | `outputs/figures/table2_e3.tex`, `calibrators.csv` |
| In text | Smoothing and source-shift results, significance statements | `e4_smoothing.csv`, `summary.csv`, `paired_differences.csv` |

## 6. Whichever option is chosen

**Do not claim**
- that the detector is over-confident or "confident but wrong";
- that calibration degrades under domain shift;
- that the rare class is the worst calibrated;
- robustness to domain shift in general. The shift tested changes source recordings only.

**Do state**
- that the results are for one small architecture, one setting per loss (γ = 2, smoothing 0.1) and three seeds;
- that the mixtures are synthetic and the mixing recipe is the same in every split;
- that the validation set is used both for early stopping and for fitting the calibrators;
- that several runs peaked at or near the last of 20 epochs;
- that ECE is biased upward on finite samples.

**Check before submission**
- Related work on focal loss and label smoothing calibration, so the under-confidence finding is positioned against results in other domains. This note does not cite specific papers; the references need to be found and verified.
- The manual listening check of sample clips, which is still outstanding.
- That any new figures are generated by a script in the repository.

## 7. Optional additions, by cost

| Addition | Why | Cost |
|---|---|---|
| Signed calibration error (confidence minus accuracy) in Table 1 | Makes "under-confident" a reported number | Minutes; saved logits |
| Coverage at thresholds on the shifted set, in the main text | Shows the threshold effect holds on unseen sources | Already computed |
| Reliability diagram for frames predicted as speech+music | Settles the rare-class question (needed for Option E) | Under an hour; saved logits |
| Focal loss with γ = 1 and γ = 5 | Shows the effect grows with γ; answers the "one setting" objection | Under 20 minutes of GPU time per setting, plus evaluation; requires a protocol change agreed by both authors |
| A harder shift condition | Needed for any robustness claim (Option D) | New data and evaluation; not feasible before the deadline |
