# Fraud Detection Modeling & Risk Decisioning — Project Plan

*My recommended version. **Status: LOCKED** (dataset: IEEE-CIS; classical ML + rules + anomaly + decisioning). Do not re-scope mid-build.*

> **What this is:** an end-to-end, single-notebook fraud detection **system** (not just a classifier) on real transaction data, built to be understood cell-by-cell and defended in an interview.
> **Why it fits your CV:** fraud detection and credit risk sit in the **same domain (Credit & Fraud Risk / CFR)** — imbalanced binary classification, point-in-time feature engineering, calibration, explainability, and cost-based decisioning. This project shows you can do all of it.

---

## 0. Guiding principles (these shape every decision below)

1. **It's a system, not a model.** Raw data → features → rules + ML + anomaly → calibrated risk score → approve/challenge/decline → monitoring. The decision at the end is the product.
2. **Teaching quality is a deliverable.** Every substantive cell is wrapped in a fixed teaching pattern (§8) so you can learn the *why*, not just run code.
3. **Leakage discipline is non-negotiable.** Fraud data is time-ordered; every historical feature is computed strictly from the past (§3, §4). This is the single most important thing that separates a real project from a Kaggle notebook.
4. **Every number is defensible.** Metrics fit the problem (PR-AUC, not accuracy), thresholds are *derived* from a cost model (not eyeballed), and model comparisons come with uncertainty (bootstrap CIs).
5. **Honest scope.** Classical ML only; no deep learning, no graph ML, no cloud/API. Cuts are documented as deliberate tradeoffs, not gaps.

---

## 1. The system at a glance

```
        RAW TRANSACTIONS (IEEE-CIS)
                 │
                 ▼
        DATA QUALITY + EDA
                 │
                 ▼
        TEMPORAL SPLIT (train / calib / valid / out-of-time test)
                 │
                 ▼
        POINT-IN-TIME FEATURE ENGINEERING
                 │
        ┌────────┼─────────────┐
        ▼        ▼             ▼
     RULES   SUPERVISED ML   ANOMALY
     ENGINE  (LR, XGB, LGBM) (Isolation Forest)
        │        │             │
        └────────┼─────────────┘
                 ▼
        EVALUATION  →  CHAMPION (XGBoost)
                 │
                 ▼
        CALIBRATION  →  RISK SCORE (0–1000)
                 │
                 ▼
        SHAP  →  REASON CODES
                 │
                 ▼
        DECISION ENGINE  →  APPROVE / CHALLENGE / DECLINE  (+ rule overrides)
                 │
                 ▼
        MONITORING (drift + performance + business KPIs)
```

---

## 2. Business framing

**Objective:** catch as much fraud as possible **while** keeping false positives (blocked good customers) low — because both missed fraud and blocked-legit customers cost money.

**Cost model (set explicit $ values up front — assumptions are fine, just state them):**

| Outcome | Cost |
|---|---|
| Missed fraud (FN) | transaction amount + chargeback fee |
| Blocked legit txn (FP, decline) | lost revenue + friction/churn cost |
| Caught fraud (TP) | savings − review cost |
| Correct approve (TN) | 0 |
| **Challenge / step-up** (the middle action) | friction cost per challenge; legit users pass with prob `p_legit_pass` (rest abandon = lost revenue); fraud rarely passes (`p_fraud_pass` low) |

The challenge economics matter because the Decision Engine has **three** actions, so the threshold search (§6) is a 2-D cost optimization, not a single cutoff.

**Constraints we design around:**
- Highly imbalanced target (~3.5% fraud) → PR-AUC-first evaluation.
- Explainability required (per-decision reason codes) → SHAP-based.
- Finite manual-review capacity → the "challenge" rate must respect a review budget.
- Low-latency scoring in spirit → we simulate per-transaction scoring and note the feature-store caveat (§7).

---

## 3. Dataset — IEEE-CIS Fraud Detection (locked)

- Kaggle "IEEE-CIS Fraud Detection" (Vesta). Join `train_transaction.csv` + `train_identity.csv` on `TransactionID`.
- ~590k transactions, ~3.5% fraud, ~394 features; only ~24% of rows have identity data (real-world messiness — becomes a feature).
- Real `TransactionDT` offset enables genuine **time-based** splitting.
- **Feature families:** `TransactionAmt`, `ProductCD`, `card1–6`, `addr1/2`, `dist1/2`, `P_/R_emaildomain`, `C1–14`, `D1–15`, `M1–9`, `V1–339`, identity `id_01–38`, `DeviceType`, `DeviceInfo`.
- **Label nuance (say this in the writeup):** `isFraud` comes from reported chargebacks, and once a card is flagged, its later transactions (~120 days) are also labeled fraud — so you're partly predicting "this card is compromised," not just "this single txn is fraud."
- **Download:** accept the competition rules on Kaggle → download all (or `kaggle competitions download -c ieee-fraud-detection`) → unzip into `data/`.

**Why this dataset over alternatives:** ULB `creditcard.csv` is PCA-anonymized (nothing to engineer); Bank Account Fraud is application-fraud (different problem, noted as a v2 direction). IEEE-CIS has interpretable-enough fields, real time ordering, and rich behavior — ideal for point-in-time feature engineering.

---

## 4. The notebook, phase by phase

This is the build order of the single end-to-end notebook. Each phase lists the **objective**, **what we do**, and the **interview point** it earns.

**Phase 0 — Setup & reproducibility.** Imports, versions, fixed seeds, config/paths. *Why:* reproducibility is an instant credibility signal.

**Phase 1 — Data quality & EDA.** Schema + data dictionary; missing-value analysis (and whether missingness predicts fraud); duplicate & validity checks; target/imbalance analysis; fraud-behavior EDA (amount fraud-vs-legit, temporal fraud rate by hour/day, fraud rate by `ProductCD`/email/device/region, `V`-block correlation). Leakage smell-test: distrust any single feature that separates fraud perfectly. *Rule:* every EDA finding must lead to a feature, a rule, or a modeling choice — no decorative plots. *Interview point:* "I understood the data before modeling it."

**Phase 2 — Temporal split & leakage doctrine.** Sort by `TransactionDT`; split **Train → Calibration → Validation → Out-of-Time (OOT) Test** in time order (never random). State the point-in-time rule that governs all of §4: for a transaction at time *T*, features use only data strictly before *T*. *Interview point:* the strongest single talking point in the project.

**Phase 3 — Feature engineering (point-in-time-safe).**
- Transaction: `log(amount)`, amount buckets, cents/decimal part, `ProductCD`, `card`/`addr` encodings.
- Temporal: hour, weekday, night flag, cyclical sin/cos(hour); use Vesta's `D1–15` (already point-in-time safe).
- Behavioral (per `card1`/email/addr, backward-looking): txn count, historical mean/median/std amount, `amount / entity_rolling_avg`, z-score vs entity history, time-since-last-txn, entity tenure (time-since-first-seen — account-age proxy; newer entities are structurally higher risk).
- Velocity (backward rolling windows): txn count & amount in trailing 5min/30min/1h/24h/7d (short windows catch bursts) + `time_since_previous_transaction`.
- Entity-region behavior (from `addr1/2`, treated as coded regions — not true geography): has this card used this region before, distinct-region count per card, `new_region` flag, deviation vs the card's usual region.
- Categorical risk encoding: frequency encoding; target encoding only via expanding-window / out-of-fold (the easiest place to leak).
- Missingness features: per-row null count, identity-match flag, `M1–9` match flags.
- **"Graph-lite" relational features** (deliberate cheap substitute for GNNs): distinct `card1` count sharing an `addr1` in a window (fraud-ring proxy), card–merchant pair novelty, compound interactions (`card1×ProductCD`) encoded by combination frequency, and **cross-field mismatch flags** (classic fraud-analyst heuristics — e.g. `P_emaildomain` vs `R_emaildomain` mismatch, or email-vs-`addr` region mismatch).
- **Cold-start rule:** unseen category → "unknown" bucket at the global prior; new entity → "new entity" flag + neutral history (never null/crash). Test it by holding out a few entities entirely.

**Phase 4 — Rules engine.** Deterministic expert rules from EDA: extreme amount vs entity baseline, high velocity, rare/new location, high-risk merchant behavior, rapid repeats. Each rule emits a **flag** (reused as an ML feature) and a **reason code**. Rules also serve as (a) a **baseline** and (b) **hard overrides** in the decision engine (e.g., blacklist/impossible-travel → hard decline). *Interview point:* real fraud systems are hybrid rules + ML; this shows you know that.

**Phase 5 — Class imbalance.** Primary: class weighting (`scale_pos_weight` / `class_weight='balanced'`). One clean comparison: SMOTE vs class weighting on XGBoost (same split, same features) reported as a single table. Golden rule: **resample train only, never validation/test.** Isolation Forest gets no imbalance treatment (unsupervised — say so, so it doesn't read as an oversight).

**Phase 6 — Models.**
- **Logistic Regression** — interpretable baseline (coefficients → odds ratios, direction, regularization).
- **XGBoost** — primary, fully tuned (time-aware CV with an **embargo gap** so rolling features don't straddle folds).
- **LightGBM** — lighter benchmark on the same framework.
- **Isolation Forest** — unsupervised anomaly score; also fed back as a feature/complementary signal, and tested for whether it adds lift over the boosted models.

**Phase 7 — Feature selection (prove the set is a decision, not "everything").** Variance threshold → multicollinearity check → model-based importance (gain/SHAP) → **stability across time folds** (drop features whose importance swings across folds — likely leakage/overfit) → **permutation / null-importance** sanity check → a final "kept/dropped & why" table.

**Phase 8 — Evaluation.** Primary: **PR-AUC / average precision**. Operational: recall @ fixed FPR budget, **Precision@K** (top-K review capacity), fraud-capture rate. A comparison table (rules-only, LR, XGBoost, LightGBM, Isolation Forest, ensemble benchmark) with **bootstrap 95% CIs** on the key gaps (if the CI crosses zero, say the models are indistinguishable). **Segment breakdown** (amount bucket, `ProductCD`, identity-present vs not) to catch hidden weak spots.
- **Champion selection by explicit rubric (not assumption):** score the candidates on (a) *predictive performance* — PR-AUC, recall@FPR, fraud capture; (b) *business performance* — expected cost, false positives / customer friction; (c) *practical factors* — inference speed, interpretability, feature stability across time. XGBoost is the expected champion (chosen for SHAP/reason-code coherence and latency), but this rubric is what **confirms** it — and documents *why* the winner won.

**Phase 9 — Calibration.** Raw boosted scores are rank-ordered but not true probabilities. Calibrate (isotonic primary, Platt compared) on the **dedicated calibration split** (never test); validate with a reliability diagram + Brier score. LR is already ~calibrated.

**Phase 10 — Explainability & reason codes.** SHAP global (what drives fraud overall) + local (why *this* txn), for the champion XGBoost. Map top SHAP contributors → human-readable reason codes (`HIGH_VELOCITY`, `UNUSUAL_AMOUNT`, `NEW_MERCHANT`, `UNUSUAL_LOCATION`). *Interview point:* mirrors real adverse-action / model-risk (SR 11-7-style) requirements.

**Phase 11 — Risk score.** Transform the calibrated probability → business-facing **0–1000 risk score** (linear, or logit-scaled to spread the distribution like a bureau score). It's a presentation layer, but it's the format fraud-ops actually use.

**Phase 12 — Decision engine & threshold optimization.** Sweep `(threshold_low, threshold_high)` over the validation set, compute **expected cost** using the §2 cost model (including challenge economics), subject to the **review-capacity budget**; pick the min-cost pair. Plot expected-cost vs threshold ("we *derived* 0.85, didn't guess it"). Apply **rule hard-overrides** on top. Output one JSON per transaction:
```json
{"fraud_probability": 0.87, "risk_score": 872, "decision": "CHALLENGE",
 "reason_codes": ["HIGH_VELOCITY", "UNUSUAL_AMOUNT", "NEW_MERCHANT"], "rule_override": null}
```

**Phase 13 — Error analysis.** Pull 20–30 false negatives and false positives, categorize failure patterns ("misses low-amount high-velocity fraud," "false-alarms on first-time merchants," "weaker when identity is missing"), cross-reference the segment breakdown, and write 3–5 concrete failure patterns + a fix hypothesis each. *Interview point:* this is what an interviewer remembers.

**Phase 14 — Monitoring (analytical, not MLOps).** Data drift (PSI / KS / JS divergence on key features, train vs OOT), prediction drift (risk-score distribution over time), performance-once-labels-arrive (PR-AUC/precision/recall), business KPIs (approve/challenge/decline rates, fraud rate, FPR). Plus a conceptual retraining trigger ("retrain when PR-AUC drops >X% or PSI exceeds a threshold") — fraud is non-stationary; this shows you know it.

**Phase 15 — Conclusion, interview kit & CV framing.** Results recap + limitations + v2 roadmap; a **~15–20 Q&A interview appendix**; and a **CV block**: 2–3 line summary + 4–6 achievement bullets + a one-line CFR positioning note.
- **Skill-progression pitch (how to tell the project's story):** frame the project as an escalating ladder of competencies, because the value isn't the number of models — it's the coherent progression: (1) statistical thinking (Logistic Regression) → (2) advanced ML (XGBoost/LightGBM) → (3) unsupervised detection (Isolation Forest) → (4) financial-domain feature engineering (velocity + behavioral + entity + temporal) → (5) imbalanced-classification discipline (PR-AUC + recall@FPR + Precision@K) → (6) risk modeling (calibration) → (7) decision science (cost-based thresholds) → (8) explainability (SHAP + reason codes) → (9) fraud operations (approve/challenge/decline) → (10) monitoring (drift + model + business KPIs). This one line turns "a collection of algorithms" into "a coherent fraud risk modeling story."

---

## 5. Modeling scope — locked

**In:** Rules engine, Logistic Regression, XGBoost (primary/champion), LightGBM (benchmark), Isolation Forest (anomaly), cost-based Decision Engine, SHAP.

**Out (documented as deliberate v2 items):** Random Forest (redundant with GBMs), any deep learning (autoencoder/LSTM/transformer), any graph ML (GNN/GraphSAGE), other anomaly methods (One-Class SVM, LOF, DBSCAN), CatBoost, live API/Docker/CI-CD, MLOps model registry, champion/**challenger** deployment, automated retraining. (Note: champion *selection* in §4 is fine — that's just picking the best model, not the A/B deployment pattern.)

---

## 6. Cross-cutting standards

- **Leakage discipline:** sorted groupby + expanding/backward windows per entity; never a global groupby; embargo gap in tuning CV.
- **Reproducibility:** fixed seeds everywhere; pinned `requirements.txt`; the notebook must run top-to-bottom on a fresh kernel with no hidden state.
- **Metrics honesty:** PR-AUC first; bootstrap CIs on comparisons; report where the model is weak, not just its best number.
- **Persisted artifact:** serialize the full fitted pipeline (encoders + calibrator + model + thresholds + rules) into a single `score(transaction) -> JSON` callable via joblib — the actual deployable object, and what powers the simulated real-time loop.

---

## 7. Tooling — deliberately lean, notebook-centric

- **Primary deliverable:** one polished `fraud_detection_end_to_end.ipynb` that runs everything top to bottom with full markdown teaching.
- **No cloud, no API, no Docker, no CI/CD** — stated as a scope decision ("serving is a natural v2 step").
- **Optional, only if it adds signal:** local MLflow for experiment tracking (nice-to-have, not required); a simple data-sanity check on load. Drift in §14 is done analytically (PSI/KS/JS) rather than pulling in heavy tooling.
- **Simulated real-time scoring:** iterate over the OOT test in `TransactionDT` order, score row-by-row through the persisted pipeline, time each pass. **Caveat to state:** in production the rolling features come from a precomputed feature store; the notebook recomputes them, so the latency figure reflects decision logic, not a production SLA.

---

## 8. Per-cell teaching format (the core of your requirement)

Every substantive cell (or tight cell group) follows this fixed pattern:

1. **Concept** — what & why, plain English
2. **Method** — the technique and why it's right here
3. **Code** — the implementation
4. **Result** — what the output/plot shows
5. **Interpretation & intuition** — meaning, intuition, common pitfalls
6. **Interview angle** — the 1–2 lines you'd actually say

Plus a closing **interview Q&A appendix** and **CV bullets** (§4, Phase 15).

---

## 9. Deliverables & repo layout

```
fraud-detection-system/
├── README.md                          # problem, architecture diagram, headline results
├── requirements.txt                   # pinned versions
├── data/                              # raw CSVs (gitignored)
├── fraud_detection_end_to_end.ipynb   # THE deliverable — full teaching notebook
├── artifacts/                         # persisted score(transaction)->JSON pipeline, plots
└── docs/
    └── model_card.md                  # intended use, data, performance, limitations, monitoring plan
```

*(A split into numbered notebooks + a `src/` package is optional and can come later; the single notebook is what you walk an interviewer through.)*

---

## 10. Suggested build order (flexible)

1. Setup + data quality + EDA + temporal split
2. Feature engineering + rules engine (+ cold-start test)
3. LR + XGBoost (tuned) + LightGBM + Isolation Forest; imbalance comparison; feature selection
4. Evaluation (PR-AUC, Precision@K, segments, bootstrap CIs) → pick champion
5. Calibration + SHAP + reason codes + risk score
6. Decision engine + cost-based thresholds + persisted artifact + simulated real-time loop
7. Error analysis + monitoring + model card + interview kit/CV bullets + final top-to-bottom clean run

---

## Why this version is strong

- A real **decisioning system** (approve/challenge/decline with reason codes), not just a classifier.
- **Leakage-safe, point-in-time** feature engineering — the #1 credibility signal in fraud/credit modeling.
- **Rules + ML + anomaly** hybrid, the way production fraud stacks actually work.
- **Calibrated** probabilities and **cost-derived** thresholds — decisions you can justify in dollars.
- **PR-AUC + Precision@K + segment breakdowns + bootstrap CIs** — honest, senior-level evaluation.
- **SHAP reason codes + model card** — explainability mapped to real model-risk practice.
- **Error analysis** — you know where and why the model breaks.
- **Built to teach:** every cell explained, plus an interview Q&A and CV bullets.
- **Honest, documented scope cuts** — maturity, not omission.

---

*Status: **LOCKED** and frozen. Say "go" (and whether you want the single notebook only) and I'll build it.*
