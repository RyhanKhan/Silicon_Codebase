# Wings R Us — Smart Last‑Minute Recommendations (WWT Unravel 2025)

**Objective**
- Recommend **Top‑3** add‑on items for each in‑progress cart to lift AOV.
- Evaluation: **Recall@3** on `test_data_question.csv` (✓ if the hidden “missing” item appears in our Top‑3).

This repository uses a **two‑stage recommender** implemented in Python/Jupyter:
1) **Stage 1 — Candidate Generation:** co‑purchase signals (P(j|i), lift, jaccard, popularity)
2) **Stage 2 — Candidate Re‑ranking:** LightGBM LambdaRank using cart, context, and behavior features

> **Note:** Our project is organized as **one folder** named **`WWT Project`**. **All files (CSVs, notebooks, artifacts, outputs) live directly in this single folder.** The notebooks therefore reference files by **filename only** (no paths).

---

## Folder Contents (single‑folder layout)

```
  # Notebooks (run in numeric order)
  1_order_data_cleaning.ipynb
  2_customer_data_cleaning.ipynb
  3_store_data_cleaning.ipynb
  4_store_data_advanced_cleaning.ipynb
  5_joining.ipynb
  6_item_level_FE.ipynb
  7_customer_level_FE.ipynb
  8_contextual_FE.ipynb
  9_merging_feature_tables.ipynb
  10_candidate_generation.ipynb
  11_candidate_re-ranking.ipynb
  12_inference.ipynb

  # Raw inputs you place in this same folder
  order_data.csv
  customer_data.csv
  store_data.csv
  test_data_question.csv

  # Stage‑1 artifacts (created by notebook 6)
  item_stats_counts_and_freq.csv
  item_cooccurrence_counts.csv
  P_j_given_i.csv
  item_lift_matrix.csv
  item_jaccard_matrix.csv
  top10_by_Pjg.json

  # Stage‑2 artifacts (created by notebook 11, or provided)
  stage2_lgbm_model.txt
  stage2_feature_columns.json
  stage2_cat_vocab.json

  # Outputs
  stage1_candidate_recommendations_top3_v2.csv
  submission_test_data_question.csv   # ✅ final submission (with spaces in headers)

  # Misc
  requirements.txt
  README.md
```

---

## Environment Setup

> Tested with **Python 3.9+**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run

### Option A — **Fast Path (no retraining)**
Reproduce the submitted file directly using the saved model & feature metadata.

1. Confirm the following files are present in **WWT Project/**:
   - `stage2_lgbm_model.txt`
   - `stage2_feature_columns.json`
   - `stage2_cat_vocab.json`
   - Stage‑1 stats: `item_stats_counts_and_freq.csv`, `item_cooccurrence_counts.csv`,
     `P_j_given_i.csv`, `item_lift_matrix.csv`, `item_jaccard_matrix.csv`, `top10_by_Pjg.json`
   - `test_data_question.csv`
2. Open **`12_inference.ipynb`** in Jupyter and **Run All**.
3. Output: **`submission_test_data_question.csv`** (in the same folder).

**Submission format (required by organizers):**
- Columns: `ORDER_ID`, `RECOMMENDATION 1`, `RECOMMENDATION 2`, `RECOMMENDATION 3` *(with spaces, by design)*
- One row per test order; no duplicates, no blanks.

### Option B — **Full Pipeline** (clean → feature engineering → train → predict)

Run notebooks **in numeric order**; each writes files back into this folder:

1. **1_order_data_cleaning.ipynb**  
   - Clean orders; explode items; make one‑hot/count columns; `total_order_price`.
   - Output: `order_data_encoded.csv`

2. **2_customer_data_cleaning.ipynb**  
   - Encode customer types → `cust_registered`, `cust_guest`, `cust_special_membership`.
   - Output: `customer_data_encoded.csv`

3. **3_store_data_cleaning.ipynb**  
   - Normalize city/state keyed by `STORE_NUMBER`.
   - Output: `store_data_cleaned.csv`

4. **4_store_data_advanced_cleaning.ipynb**  
   - One‑hot `city_*`, drop postal code.
   - Output: `store_data_encoded.csv`

5. **5_joining.ipynb**  
   - Join orders + customers + stores → master dataset.
   - Output: `master_dataset.csv`

6. **6_item_level_FE.ipynb**  
   - Build co‑purchase features:
     - `item_stats_counts_and_freq.csv`
     - `item_cooccurrence_counts.csv`
     - `P_j_given_i.csv`
     - `item_lift_matrix.csv`
     - `item_jaccard_matrix.csv`
     - `top10_by_Pjg.json`

7. **7_customer_level_FE.ipynb**  
   - Per‑customer aggregates:
     - `orders_count`, `items_count`, `repeat_purchase_rate`, `avg_order_value`
     - `weekend_order_ratio`, `most_common_order_hour`, `most_common_order_dow`
     - `store_diversity_count`
   - Output: `customer_level_features.csv`

8. **8_contextual_FE.ipynb**  
   - Time/store patterns; enrich with `store_STATE` + store city one‑hots.
   - Output: `contextual_features.csv`

9. **9_merging_feature_tables.ipynb**  
   - Merge master + customer + context → `modeling_dataset.csv`

10. **10_candidate_generation.ipynb**  
    - Stage‑1 scoring per cart using P(j|i), lift, jaccard, popularity (+ diversity rule).
    - Output (for inspection): `stage1_candidate_recommendations_top3_v2.csv`

11. **11_candidate_re-ranking.ipynb**  
    - Leave‑One‑Out training pairs: 50 candidates/cart, 1 positive.
    - LightGBM LambdaRank; time‑based split; early stopping on NDCG@3.
    - Artifacts written: `stage2_lgbm_model.txt`, `stage2_feature_columns.json`, `stage2_cat_vocab.json`
    - Validation (our run):
      - **NDCG@3 ≈ 0.716**
      - **Recall@1 ≈ 0.64**, **Recall@3 ≈ 0.77**

12. **12_inference.ipynb**  
    - Score `test_data_question.csv` with Stage‑2 model.
    - Enforce unique Top‑3 and avoid recommending items already in the cart.
    - **Final:** `submission_test_data_question.csv`

---

## Design Choices

- **Two‑Stage** for accuracy + speed: high‑recall candidates, then a ranker optimized for NDCG@3/Recall@3.
- **Time‑based validation** to mimic production drift.
- **Consistent category encoding** saved to `stage2_cat_vocab.json` and reused at inference.
- **Safety checks** at inference: no duplicates within Top‑3; no items already in the cart; all SKUs exist.

---

## Troubleshooting

- **Memory during training (Notebook 11):**  
  Use chunked reads, `float32` downcast, `joblib.Parallel(backend="threading")`.  
  If needed, reduce `MAX_TRAIN_BASKETS`, `NEG_PER_CART`, or use **Option A** (Fast Path).

- **Category codes mismatch:**  
  Ensure `stage2_cat_vocab.json` exists; otherwise inference falls back to test‑only codes (slightly lower accuracy).

- **Submission headers:**  
  Must be exactly `RECOMMENDATION 1/2/3` (with spaces). The inference notebook enforces this.

---

## Results Summary

- Internal validation: **Recall@3 ≈ 0.7695**, **Recall@1 ≈ 0.6397**, **NDCG@3 ≈ 0.716**  
- Typical Top‑3 aligns with co‑purchase patterns (e.g., wings → dips/fries add‑ons).

---

- ## Cloning this repo (Git LFS required)

This repo uses **Git LFS** for large CSVs (e.g., `order_data.csv`). Before cloning:

```bash
git lfs install          # one-time on your machine
git clone <YOUR-REPO-URL>
cd <your-repo>
git lfs pull             # fetch the large files (if they didn’t auto-download)

