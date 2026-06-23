# My Lesson — Regression Metrics: MAE, MAPE & R²

> A personal reference for the three metrics I use to judge regression models
> (models that predict a continuous number, like price, demand, or temperature).

---

## The setup

For each row in my test set I have:

- **yᵢ** — the actual/true value
- **ŷᵢ** — the value my model predicted
- **ȳ** — the mean of all the actual values
- **n** — the number of rows

The error for one row is simply `yᵢ − ŷᵢ`.

---

## 1. MAE — Mean Absolute Error

**Formula**

```
MAE = (1/n) · Σ |yᵢ − ŷᵢ|
```

- The **average absolute size** of my errors.
- Same **units** as the target. If I'm predicting dollars, MAE is in dollars.
- **Robust to outliers** (it doesn't square the errors like RMSE does).

> *"On average, my prediction is off by ±MAE."*

Lower is better. `0` = perfect.

---

## 2. MAPE — Mean Absolute Percentage Error

**Formula**

```
MAPE = (100% / n) · Σ |yᵢ − ŷᵢ| / |yᵢ|
```

- Expresses each error as a **percentage of the true value**, then averages.
- **Unit-free** — lets me compare models across datasets of different scales.
- ⚠️ **Breaks when yᵢ = 0** (division by zero) and over-penalises under-prediction.

> *"On average, my prediction is off by MAPE percent."*

Lower is better. A MAPE of 8% means typical error ≈ 8% of the actual value.

---

## 3. R² — Coefficient of Determination

**Formula**

```
        SS_res     Σ (yᵢ − ŷᵢ)²
R² = 1 − ────── = 1 − ────────────
        SS_tot     Σ (yᵢ − ȳ)²
```

- The **fraction of variance** in the target that my model explains.
- A baseline that always predicts the mean ȳ scores **R² = 0**.
- A perfect model scores **R² = 1**. A model *worse than the mean* can go **negative**.

> *"My model explains R²×100% of the variation in the data."*

---

## Picking the right one

| Question I'm asking | Metric |
|---|---|
| "How many units/dollars am I typically off by?" | **MAE** |
| "What's my error as a percentage?" (stakeholder-friendly) | **MAPE** |
| "How much better than a naive mean-guess is my model?" | **R²** |

**Rule of thumb:** report MAE *and* R² together — MAE gives the human-readable
magnitude, R² gives the "is this model worth anything" sanity check. Add MAPE
when the audience thinks in percentages and the target is never near zero.

---

## In scikit-learn

```python
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)   # returns a fraction; ×100 for %
r2   = r2_score(y_true, y_pred)
```

> See the interactive demo in `lesson/webapp/index.html` — type in actual vs.
> predicted values and watch all three metrics update live.

---

## Case study — feature engineering on the HDB resale model

Same Linear Regression, same 233,479 real records, same 80/20 split. Only the
**features** change (Part B vs Part B+ from `C32_practice.ipynb`):

| Metric | Part B (5 features) | Part B+ Enhanced (8 features) | Improvement |
|---|---|---|---|
| **MAE** | S$82,761 | **S$52,882** | ↓ S$29,879 (~36%) |
| **MAPE** | 16.7% | **10.9%** | ↓ 5.8 pts |
| **R²** | 0.704 | **0.868** | ↑ +0.164 |

The three added columns: `remaining_lease_years` (cleaned from text — the heavy
lifter, since lease decay drives HDB price), `txn_year` (market trend over time),
and `flat_model`. This 8-feature *linear* model even beats the *Random Forest*
on the weaker 5-feature set (~S$73k MAE).

**Lesson:** better features often beat a fancier model. See the "Part B vs
Enhanced" tab in the webapp.

---

## Case study — model choice & overfitting (Random Forest vs Gradient Boosting)

Now features are **fixed** (8-feature set) and the **model** changes. Test-set scores:

| Metric (TEST) | Random Forest | Gradient Boosting |
|---|---|---|
| MAE | **S$25,769** | S$32,499 |
| MAPE | **4.9%** | 6.2% |
| R² | **0.962** | 0.946 |

But the **overfitting gap** (train R² − test R²) tells the real story:

| Model | TRAIN R² | TEST R² | Gap |
|---|---|---|---|
| Random Forest | 0.994 | 0.962 | **+0.032** (large — memorising) |
| Gradient Boosting | 0.947 | 0.946 | **+0.001** (tiny — honest) |

**Lesson:** "better" isn't only the lowest test number. Random Forest wins raw
error here, but Gradient Boosting barely overfits, so it's more trustworthy as
data drifts. Always judge by the **test** score *and* the train↔test gap. See the
"Part C: RF vs Boosting" tab in the webapp.

---

## Case study — stacking: does a *team* of models win? (Part D vs Part D+)

**Stacking** trains several base models side by side, then a small "manager"
model learns how to blend their predictions. Same 8-feature set, test scores:

| Model | TEST MAE | TEST MAPE | TEST R² |
|---|---|---|---|
| Tuned Gradient Boosting (best single) | S$26,439 | 5.1% | 0.964 |
| Stack — Part D (RF + default GB → Linear) | S$25,732 | 4.9% | 0.962 |
| **Stack+ — Part D+ (RF + tuned HGB + Ridge → Ridge, passthrough)** | **S$24,630** | **4.7%** | **0.967** |

**Why the enhanced stack wins:** the Part D team barely helped because both bases
were trees and the GB was untuned — too *similar*. Stacking rewards **diversity**.
Adding a tuned booster *and* a non-tree learner (`RidgeCV`), plus a regularised
manager that also sees the raw features (`passthrough=True`), finally beats every
single model — it's the best result in the whole lab.

**Lesson:** stacking *can* beat the best single model, but only when the team is
well-designed (strong + diverse bases, regularised manager). The gain here is real
yet modest (~S$1,800 lower MAE) for ~3× the training cost — so reach for it only
when that last sliver of accuracy pays off. A single tuned Gradient Boosting gives
~95% of the benefit far more cheaply. See the "Part D: Stacking" tab in the webapp.

---

## The leverage ladder (the whole lab in one view)

| Step | Change | MAE | R² |
|---|---|---|---|
| Baseline | 3 features, Linear | S$102k | ~0.50 |
| Part B | +flat_type, town | S$82.8k | 0.704 |
| Part B+ | +3 engineered features | S$52.9k | 0.868 |
| Part C | Random Forest | S$25.8k | 0.962 |
| Part C+ tuned | Tuned Gradient Boosting | S$26.4k | 0.964 |
| **Part D+** | **Enhanced stacking** | **S$24.6k** | **0.967** |

Biggest leverage = **features** (S$102k → S$53k), then a **tuned model**
(S$53k → S$26k), then a **well-designed ensemble** for the final sliver.
