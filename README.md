# Movie & Show Success Predictor

Predicting whether a movie or TV show on Amazon is "good" (IMDb score of 7 or higher) using only the information available about a title.

**Author:** Aman Verma

---

## Overview

This project merges two datasets (titles and credits), cleans them, and trains two classification models to predict if a title will be rated 7 or above on IMDb. It also checks how much the results change when columns that are only known *after* release are removed, so the final score is honest and not inflated by data leakage.

An interactive dashboard presents the results, models and data insights.

## Dataset

| File | Description |
|---|---|
| `titles.csv` | One row per title: type, description, release year, runtime, genres, production countries, seasons, IMDb score and votes, TMDB popularity and score |
| `credits.csv` | One row per actor or director of a title |

After merging on `id`, there are **118,127 rows** but only **8,156 unique titles**, because each title appears once for every actor or director. About **23%** of the titles are rated "good", so the classes are imbalanced.

> Add the link to the dataset source here: `[dataset link]`

## Approach

1. **Merge** credits and titles on `id`.
2. **Clean:**
   - dropped rows where `imdb_score` (the target) is missing
   - `seasons` set to 0 for movies
   - remaining missing numbers filled with the median, using training data only
   - removed duplicate rows
3. **EDA:** distributions, outliers, top genres and countries, correlation heatmap.
4. **Feature engineering:** first genre and first country, one-hot encoded (using `ast.literal_eval` to parse the list strings safely).
5. **Split by title, not by row:** `StratifiedGroupKFold` keeps every row of a title in the same set, so no title appears in both train and test.
6. **Models:** Logistic Regression and XGBoost, both with class-imbalance handling.
7. **Validation:** 5-fold grouped cross-validation, plus a leakage check that removes post-release columns.

## Results

### With all features (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8297 | 0.5842 | 0.8113 | 0.6793 | 0.8960 |
| XGBoost | 0.8741 | 0.6961 | 0.7694 | 0.7309 | 0.9274 |

Grouped 5-fold CV ROC-AUC: Logistic Regression **0.8848** (± 0.0137), XGBoost **0.9401** (± 0.0065).

### Leakage check: without `tmdb_score`, `imdb_votes`, `tmdb_popularity`

These columns are only known after a title is released, so the honest result is the one below.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7366 | 0.5326 | 0.7848 |
| XGBoost | 0.7584 | 0.5654 | **0.8078** |

Evaluated on unique test titles only (1,632 titles), XGBoost reaches ROC-AUC **0.816**, F1 0.575 and accuracy 0.753.

### Key findings

- XGBoost beats Logistic Regression in every comparison.
- Removing post-release columns lowers XGBoost ROC-AUC from 0.93 to 0.81, which shows how much leakage inflates the score.
- TV shows are rated much higher than movies: about 64% of shows are "good" compared with about 19% of movies, so `seasons` is the most important pre-release feature.
- Genre matters a lot: documentaries are "good" about 63% of the time, while horror is only about 2%.

## Dashboard

An interactive dashboard (dark Cinema theme) with three tabs:

- **Overview:** headline numbers and the leakage effect
- **Models:** switch between all features and pre-release features, compare metrics and see top features
- **Data insights:** score cut-off slider, genre tiles, and "good" share by type, genre, country and decade

Dashboard link: `[add your dashboard link here]`

## Limitations

- The same title still appears in many rows (one per actor or director), so rows are not fully independent. Splitting by title id avoids leakage, but test results are slightly weighted towards titles with large casts.
- All features are title-level. The cast and director information in `credits.csv` is not used by the model yet.
- Only the first genre and first country of each title are used.
- Imputation for cross-validation was done once on the training set, not separately inside each fold.
- IMDb scores come from people's opinions, so part of the score cannot be predicted from these columns.

## Future work

- Add features from credits: number of cast members, director experience, average actor experience.
- Use all genres and countries (multi-hot encoding) and TF-IDF on the description.
- Tune hyperparameters (for example with Optuna) and add SHAP explanations.
- Build a content-based recommender from descriptions and genres.
- Turn the dashboard into a Streamlit app with live predictions.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

1. Place `credits.csv` and `titles.csv` in the same folder as the notebook.
2. Open `code_corrected.ipynb` and make sure the two `pd.read_csv(...)` paths point to those files.
3. Run all cells.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter.
