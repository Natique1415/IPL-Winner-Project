# IPL Match Winner Classification : DMDW Group Project

## Project overview

This project uses three classification algorithms covered in our Data Mining and Data Warehousing (DMDW) coursework to predict the winner of an Indian Premier League (IPL) match:

- Decision Tree
- Categorical Naive Bayes
- K-Nearest Neighbours (KNN)

The prediction is made **after the toss but before the first ball**. The target, `team1_won`, is `1` when the team listed as `team1` wins and `0` when the team listed as `team2` wins.

The project combines two kinds of historical information: match-level data from `matches.csv` and ball-by-ball data from `deliveries.csv`. A shared feature-engineering module creates the same 11 pre-match features for all three models, helping make their results comparable. The models are evaluated on later matches that were not used to choose their settings.

**Main finding:** Decision Tree performs best among the three models on the current held-out test set. However, none of the models beats the majority-class baseline in accuracy. The report should present this limitation clearly rather than claiming that the models are reliable match predictors.

For Git, virtual-environment, and dependency setup instructions, use [`installation.md`](installation.md). This README focuses on the project, methodology, results, and material useful for the presentation and report.

---

## Dataset and project files

### Dataset source

The project uses the [IPL Complete Dataset by `patrickb1912` on Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020/). Although the URL mentions 2008–2020, the supplied local dataset covers seasons from **2008 through 2024**.

| File | What it contains | Raw size |
|---|---|---:|
| `matches.csv` | One row per match: date, participating teams, toss information, venue, winner, and result details | 1,095 rows × 20 columns |
| `deliveries.csv` | Ball-by-ball information: match ID, batting and bowling teams, runs, extras, and wickets | 260,920 rows × 17 columns |

The datasets are connected by `matches.csv` column `id` and `deliveries.csv` column `match_id`. In the supplied data, all 1,095 match IDs have delivery records, and the delivery data does not contain unknown match IDs.

### Main project files

```text
IPL-Winner-Project/
├── matches.csv
├── deliveries.csv
├── requirements.txt
├── installation.md
├── README.md
├── shared_features.py
├── model_utils.py
├── Step-2/
│   ├── viewing-data.py
│   └── team-renaming.py
├── Step-3/
│   └── cleaning-data.py
├── Step-4/
│   └── Classification/
│       └── decision-tree.py
├── Step-5/
│   └── Naive-Bayes/
│       └── nb.py
├── Step-6/
│   └── KNN/
│       └── knn.py
└── Step-7/
    └── Model-Comparison/
        ├── compare-models.py
        ├── decision_tree_metrics.json
        ├── naive_bayes_metrics.json
        ├── knn_metrics.json
        ├── decision_tree_metrics.png
        ├── decision_tree_structure.png
        ├── naive_bayes_metrics.png
        ├── knn_metrics.png
        ├── knn_nearest_neighbors.png
        └── classification_comparison.png
```

`shared_features.py` prepares a consistent set of pre-match features. `model_utils.py` contains shared support for chronological validation and reporting. Each classifier saves a JSON file with its metrics and model-selection details. The generated PNGs provide visual material for explaining each model and comparing their performance.

---

## Data and methodology shared across the project

This section describes concepts that each person presenting a work area should understand, even if they are not responsible for implementing every part.

### Prediction target and timing

The model predicts the winner using information available **after the toss but before the first ball**. Toss information is allowed under this prediction setup. The model must not use the current match's final score, winner, result margin, Player of the Match, or any other information that would only become known after the match.

| Target value | Meaning |
|---:|---|
| `1` | Team 1 wins |
| `0` | Team 2 wins |

“Team 1” and “Team 2” refer to the positions in each row of `matches.csv`; they are not fixed franchise names.

### Shared feature-engineering module

All three classifiers use `shared_features.py` to create the same 11 features. This avoids each algorithm independently calculating historical records in a slightly different way.

| Feature | Meaning in simple language | Information source |
|---|---|---|
| `team1_won_toss` | Whether Team 1 won the toss | `matches.csv` |
| `team1_bats_first` | Whether Team 1 bats first based on the toss winner's decision | `matches.csv` |
| `prior_win_rate_diff` | Team 1's overall historical win rate minus Team 2's | Earlier matches |
| `recent_5_win_rate_diff` | Difference between the teams' win rates over their previous five matches | Earlier matches |
| `recent_10_win_rate_diff` | Difference between the teams' win rates over their previous ten matches | Earlier matches |
| `head_to_head_win_rate_diff` | Team 1's historical win rate against Team 2, centred around 0.5 | Earlier matches between the pair |
| `venue_win_rate_diff` | Difference between the teams' historical win rates at the current venue | Earlier matches at that venue |
| `avg_runs_scored_diff` | Difference in the teams' historical average runs scored | Earlier deliveries, summarised by match and team |
| `avg_runs_conceded_diff` | Difference in historical average runs conceded, oriented so a positive value favours Team 1 | Earlier deliveries, summarised by match and team |
| `avg_wickets_taken_diff` | Difference in the teams' historical average wickets taken | Earlier deliveries, summarised by match and team |
| `avg_wickets_lost_diff` | Difference in historical average wickets lost, oriented so a positive value favours Team 1 | Earlier deliveries, summarised by match and team |

Some run and wicket differences are divided by constants to keep their numeric scales manageable. These scaled values are not percentages or probabilities.

### How the ball-by-ball data is used

`deliveries.csv` contains multiple rows per match—typically one for each delivery—so it cannot be added directly to `matches.csv` as if both files had one row per match. The feature pipeline aggregates delivery records by match and team to calculate summaries such as runs scored, runs conceded, wickets taken, and wickets lost. These match-level summaries then become historical records for future predictions.

The delivery columns used by the shared feature pipeline include `match_id`, `batting_team`, `bowling_team`, `total_runs`, `is_wicket`, and `extras_type`. Historical team names are standardised before the summaries are joined to the match history.

**Important:** deliveries from the match currently being predicted are not used as predictors for that match. They are used only to update team history after that match has taken place.

### Avoiding future-data leakage

**Data leakage** occurs when model inputs contain information that would not have been available at the time of prediction. It can make a model look more successful than it really is.

The shared pipeline addresses this by:

1. Sorting match records chronologically.
2. Building features for every match on a given date before adding results from that date to historical records. This is a conservative rule when multiple matches share a date.
3. Using only earlier matches for rolling form, head-to-head history, venue statistics, and delivery-derived performance averages.
4. Updating historical records after features for the current date have been made.
5. Excluding post-match result fields from the model features.

### Train, validation, and test design

The models are evaluated chronologically because a real prediction system would learn from past seasons and make predictions on later matches. A random split could mix future matches into training and give an unrealistic estimate of performance.

| Portion | Matches | Purpose |
|---|---:|---|
| Earlier period | 888 | Hyperparameter selection and final model fitting |
| Latest held-out period | 202 | Final evaluation only |
| **Total usable matches** | **1,090** | |

The test period is **9 April 2022 to 26 May 2024**. The earlier 888 matches are used in three expanding-window chronological validation folds. Every fold trains on earlier dates and validates on later dates; matches on the same date are kept on the same side of a split.

The best settings are selected using the **mean validation balanced accuracy**. The selected model is then refitted using all 888 pre-test matches and evaluated once on the 202 held-out matches. The final test set must not be used to tune parameters.

---

## Data collection and exploration (Steps 1–2)

### Purpose

This part explains where the data came from and whether the raw files are suitable for analysis. It establishes what each dataset contains before cleaning or modelling begins.

### Step 1 : Data collection

The Kaggle IPL dataset provides two complementary files:

- `matches.csv` provides one record per match, including participating teams, toss details, venue, date, and winner.
- `deliveries.csv` provides the ball-by-ball records needed to summarise historical batting and bowling performance.

The source files are retained as inputs. Scripts use in-memory copies and should not overwrite the original data.

### Step 2 : Data understanding and exploration

**Script:** `Step-2/viewing-data.py`

The script inspects both CSV files and reports dimensions, column names, missing values, data types, duplicate rows, match-ID relationships, and team appearances. This helps answer questions such as:

- How many rows and columns are present in each file?
- Which columns have missing values, and are those fields used by the classifiers?
- Does every match have delivery records?
- Are any delivery records associated with a match ID absent from `matches.csv`?
- Do the raw files use different names for the same historical franchise?

**Script:** `Step-2/team-renaming.py`

This script counts occurrences of raw team names and lists the seasons in which each name occurs. It is an exploration tool for identifying naming changes; the shared feature pipeline applies the actual standardisation used for model features.

### Key observations

- `matches.csv`: 1,095 rows and 20 columns.
- `deliveries.csv`: 260,920 rows and 17 columns.
- Neither raw dataset contains exact duplicate rows in the supplied version.
- Five matches have no known winner. Those matches cannot provide a supervised target for winner classification.
- All 1,095 match IDs are represented in the delivery data.
- Some descriptive or result-related fields are missing. A missing field does not automatically make a row unusable; whether it matters depends on how the field is used.

### Presentation/report tips

Explain why the project needs both files, and distinguish a **match-level dataset** from a **ball-by-ball dataset**. Show one or two examples of missing-value or team-name findings rather than reading every column from the terminal. Be ready to explain that exploration identifies issues; it does not itself train a classifier.

---

## Data preprocessing (Step 3)

### Purpose

Preprocessing prepares the raw information for reliable analysis. The most important tasks for this project are checking duplicate rows, handling matches without known winners, and giving historical franchises consistent names.

**Script:** `Step-3/cleaning-data.py`

The cleaning script reports duplicate counts, applies franchise-name mappings to in-memory copies of both datasets, removes duplicate rows from those copies, and identifies matches with known winners. It **does not write cleaned CSV files**. The classifiers load the raw data and apply the same necessary preparation through `shared_features.py`.

### Franchise-name standardisation

A franchise can have different names in different seasons. If these names are not standardised, the feature pipeline could treat one franchise as multiple teams and calculate incorrect historical win rates or performance averages.

| Raw/historical name | Standardised name |
|---|---|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |
| Rising Pune Supergiants | Rising Pune Supergiant |

The names are mapped in the match data columns `team1`, `team2`, `toss_winner`, and `winner`, and in the delivery data columns `batting_team` and `bowling_team`.

Different franchises remain separate. For example, Deccan Chargers is not merged with Sunrisers Hyderabad, and Gujarat Lions is not merged with Gujarat Titans. Renaming history is not the same as treating two distinct franchises as the same team.

### Filtering for classification

A supervised classifier needs a known outcome label. The shared feature pipeline retains 1,090 usable matches after excluding matches with missing required pre-match information, invalid dates or toss decisions, unknown winners, or a winner who is not one of the two participating teams. The exact filter is designed for classification; it does not mean the excluded match never occurred.

The delivery data is kept available for historical summaries. Individual delivery rows do not become separate winner-prediction samples; the model still predicts one winner per match.

### Presentation/report tips

Explain why a consistent franchise name matters to historical statistics. Also make clear that the cleaning script is an audit/preprocessing demonstration: it reports its work without saving replacement CSVs. The shared feature module performs the preparation required when the classifiers build their inputs.

---

## Decision Tree classification (Step 4)

**Script:** `Step-4/Classification/decision-tree.py`

### How the algorithm works

A Decision Tree predicts an outcome by asking a sequence of questions about the input features. Each internal node tests a feature threshold, each branch represents the answer to that test, and each leaf provides the predicted class.

This model uses `criterion="entropy"`, linked to **Information Gain** from the course. Entropy measures how mixed the two outcome classes are at a node. A split is useful when it reduces that uncertainty.

### Model inputs and tuning

The Decision Tree uses the 11 shared pre-match features described above. Candidate settings are compared across chronological validation folds, including:

- `max_depth`: how many levels deep the tree can grow.
- `min_samples_leaf`: the minimum number of training samples in a leaf.
- `min_samples_split`: the minimum number of samples required before a node can split.
- `class_weight`: whether class weights are left unchanged or balanced.

The settings with the highest mean validation balanced accuracy are selected. The final tree is then fitted on the 888 earlier matches.

### Selected settings

- `criterion`: `entropy`
- `max_depth`: `3`
- `min_samples_leaf`: `20`
- `min_samples_split`: `2`
- `class_weight`: `None`
- `random_state`: `42`
- Mean validation balanced accuracy: **54.5%**

### Visual outputs

- `decision_tree_structure.png`: simplified tree diagram with plain-language questions, Yes/No branches, and coloured prediction leaves.
- `decision_tree_metrics.png`: test confusion matrix and chart of the six evaluation metrics.
- `decision_tree_metrics.json`: test metrics and model-selection details.

The tree diagram is intentionally simplified to be easier to present. It shows up to three decision levels; where additional branches are hidden, the visualization uses the subtree's prediction rather than displaying every internal node. Counts shown in the tree are training-node counts, not the test confusion matrix.

### Current test results

| Metric | Result |
|---|---:|
| Accuracy | **50.0%** |
| Precision | **54.4%** |
| Recall | **34.6%** |
| Specificity | **67.4%** |
| F1-score | **42.3%** |
| Balanced accuracy | **51.0%** |

Confusion matrix, with rows as actual classes and columns as predicted classes in the order **Team 2 wins, Team 1 wins**:

| Actual outcome | Predicted Team 2 | Predicted Team 1 |
|---|---:|---:|
| Team 2 wins | 64 | 31 |
| Team 1 wins | 70 | 37 |

### Presentation/report tips

Explain one path through the tree as if answering a sequence of questions. Define entropy and Information Gain briefly; do not imply that the tree has discovered causal relationships. The main result is that Decision Tree performs best among these three tested models, but its accuracy remains below the majority-class baseline.

---

## Categorical Naive Bayes classification (Step 5)

**Script:** `Step-5/Naive-Bayes/nb.py`

### How the algorithm works

Naive Bayes estimates which outcome class is more likely after considering the observed feature values. It applies Bayes' theorem and makes a simplifying assumption that the features are conditionally independent given the class. This assumption is useful computationally, but real match features can be related to one another.

The categorical version expects categories rather than arbitrary continuous numbers. The script therefore discretizes numeric feature values into bins before estimating the class-conditional probabilities.

### Binning and smoothing

Two main settings are selected through chronological validation:

- `binning`: the coarse or fine feature-binning scheme.
- `alpha`: the Laplace-smoothing value.

**Laplace smoothing** adds a small amount to category counts so that a category not seen for a class does not force its estimated probability to zero. The selected settings are those with the highest mean validation balanced accuracy.

### Selected settings

- `alpha`: `0.1`
- `binning`: `coarse`
- `min_categories`: `8`
- Mean validation balanced accuracy: **55.3%**

### Visual outputs

- `naive_bayes_metrics.png`: test confusion matrix and six-metric chart.
- `naive_bayes_metrics.json`: test metrics and tuning details.

### Current test results

| Metric | Result |
|---|---:|
| Accuracy | **43.6%** |
| Precision | **42.9%** |
| Recall | **19.6%** |
| Specificity | **70.5%** |
| F1-score | **26.9%** |
| Balanced accuracy | **45.1%** |

Confusion matrix, with rows as actual classes and columns as predicted classes in the order **Team 2 wins, Team 1 wins**:

| Actual outcome | Predicted Team 2 | Predicted Team 1 |
|---|---:|---:|
| Team 2 wins | 67 | 28 |
| Team 1 wins | 86 | 21 |

### Interpretation and presentation/report tips

Naive Bayes has the highest specificity of the three models, but its low recall means it correctly identifies only a small fraction of Team 1 wins. It predicts Team 2 more often than the actual class distribution would suggest. Explain conditional probabilities and smoothing, and be honest that this model's test performance is weak despite its validation score.

---

## K-Nearest Neighbours classification (Step 6)

**Script:** `Step-6/KNN/knn.py`

### How the algorithm works

KNN predicts a match by comparing it with labelled matches in the training data. It finds the `K` nearest training examples in feature space and uses their classes to determine the prediction.

A match has 11 numeric features with different ranges. KNN therefore uses `StandardScaler` to put the features on comparable scales. The scaler is fitted within each training fold only, preventing information from validation or test observations from influencing the scaling parameters.

### Tuning and selected settings

The tuning process compares values of `n_neighbors`, voting weights, and distance metrics. The selected settings are:

- `n_neighbors`: `21`
- `weights`: `uniform`
- `p`: `1`, which gives Manhattan distance
- Mean validation balanced accuracy: **52.7%**

With uniform voting, all 21 neighbours have equal voting weight. Distance-weighted voting would give closer neighbours more influence; it was also considered during tuning.

### Visual outputs

- `knn_metrics.png`: test confusion matrix and six-metric chart.
- `knn_nearest_neighbors.png`: a test match plotted alongside its selected neighbours.
- `knn_metrics.json`: test metrics and tuning details.

The nearest-neighbours plot displays only two features so the figure is readable. The neighbours themselves are selected using the full 11-feature, scaled space. The 2D illustration should therefore not be interpreted as the exact distance geometry used by KNN.

### Current test results

| Metric | Result |
|---|---:|
| Accuracy | **44.6%** |
| Precision | **46.4%** |
| Recall | **29.9%** |
| Specificity | **61.1%** |
| F1-score | **36.4%** |
| Balanced accuracy | **45.5%** |

Confusion matrix, with rows as actual classes and columns as predicted classes in the order **Team 2 wins, Team 1 wins**:

| Actual outcome | Predicted Team 2 | Predicted Team 1 |
|---|---:|---:|
| Team 2 wins | 58 | 37 |
| Team 1 wins | 75 | 32 |

### Presentation/report tips

Use the neighbours plot to explain how similarity-based classification works. Mention why scaling matters for distance calculations and why a two-dimensional illustration cannot show the full 11-dimensional distance used by the model.

---

## Model comparison and findings (Step 7)

**Script:** `Step-7/Model-Comparison/compare-models.py`

This script reads the three JSON metric files, prints the comparable test metrics as percentages, and generates the grouped bar chart `classification_comparison.png`. It creates **only the PNG comparison chart**; a CSV output is not required.

### Test-set comparison

All models were evaluated on the same 202 matches (9 April 2022–26 May 2024).

| Metric | Decision Tree | Naive Bayes | KNN |
|---|---:|---:|---:|
| Accuracy | **50.0%** | 43.6% | 44.6% |
| Precision | **54.4%** | 42.9% | 46.4% |
| Recall | **34.6%** | 19.6% | 29.9% |
| Specificity | 67.4% | **70.5%** | 61.1% |
| F1-score | **42.3%** | 26.9% | 36.4% |
| Balanced accuracy | **51.0%** | 45.1% | 45.5% |
| Mean validation balanced accuracy | 54.5% | **55.3%** | 52.7% |

### Confusion matrices in one table

The positive class is `team1_won = 1`. For each model, TN means an actual Team 2 win predicted as Team 2, FP means an actual Team 2 win predicted as Team 1, FN means an actual Team 1 win predicted as Team 2, and TP means an actual Team 1 win predicted as Team 1.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Decision Tree | 64 | 31 | 70 | 37 |
| Naive Bayes | 67 | 28 | 86 | 21 |
| KNN | 58 | 37 | 75 | 32 |

### Majority-class baseline

The 888 pre-test matches contain 448 Team 1 wins, making Team 1 the majority class in the training data. A simple baseline that always predicts Team 1 therefore achieves **107 correct predictions out of 202 test matches, or approximately 53.0% accuracy**.

None of the classifiers beats this baseline in accuracy. Decision Tree has the highest accuracy and balanced accuracy among the three models, and Naive Bayes has the highest specificity. However, the Decision Tree's balanced accuracy of 51.0% is only slightly above 50%, and none of the models demonstrates strong predictive performance on this test period.

The mean validation balanced accuracies (52.7%–55.3%) are higher than the test balanced accuracies (45.1%–51.0%). That gap shows that model-selection performance did not reliably transfer to the later matches, especially for Naive Bayes. This is a reason to be cautious about generalising from validation scores.

### Generated output files

| File | Purpose |
|---|---|
| `decision_tree_metrics.json` | Decision Tree metrics and selected settings |
| `naive_bayes_metrics.json` | Naive Bayes metrics and selected settings |
| `knn_metrics.json` | KNN metrics and selected settings |
| `decision_tree_structure.png` | Simplified Decision Tree diagram |
| `decision_tree_metrics.png` | Decision Tree confusion matrix and metrics |
| `naive_bayes_metrics.png` | Naive Bayes confusion matrix and metrics |
| `knn_metrics.png` | KNN confusion matrix and metrics |
| `knn_nearest_neighbors.png` | KNN nearest-neighbours illustration |
| `classification_comparison.png` | Grouped metric comparison across all models |

### Overall conclusion

The Decision Tree is the best of the three models on this particular held-out test set, but it does not beat the majority baseline. Naive Bayes has a higher specificity but very low recall; KNN sits between the other two on several metrics. The result is not that one algorithm solves IPL match prediction, but that these models and available features have limited predictive power on the unseen period. The presentation should explain both the models' strengths and this limitation.

### Presentation/report tips

Keep the comparison focused on the same test matches. Do not call the highest-scoring model “accurate” without qualification, and do not claim that tuning improved prediction unless a controlled experiment supports that claim. The useful story is the complete workflow: historical feature creation, time-aware evaluation, transparent comparison, and a realistic interpretation of limitations.

---

## Evaluation metrics: quick reference

Team 1 wins (`team1_won = 1`) is treated as the **positive class** throughout the metric tables.

| Metric | Meaning |
|---|---|
| **Accuracy** | Fraction of all test matches predicted correctly |
| **Precision** | Among matches predicted as Team 1 wins, fraction that actually were Team 1 wins |
| **Recall** | Among actual Team 1 wins, fraction correctly identified |
| **Specificity** | Among actual Team 2 wins, fraction correctly identified |
| **F1-score** | Harmonic mean of precision and recall; becomes low when either is low |
| **Balanced accuracy** | Average of recall and specificity, giving both classes equal weight |

Accuracy is easy to understand but can hide a model's tendency to favour one class. Recall and specificity show how well each class is identified. F1-score summarises precision and recall. Balanced accuracy is used to choose model settings because both outcomes matter.

---

## Limitations and notes for the report

1. **Limited predictive signal:** IPL outcomes depend on factors not fully represented by the historical team-level features, including playing XI, player availability, injuries, pitch conditions, weather, and in-game events.
2. **Limited test sample:** the final evaluation uses 202 matches. A difference of a few correct predictions changes the reported percentage noticeably.
3. **Validation-to-test gap:** selected settings score better on validation than on the final test set. Validation performance is not a guarantee of future performance.
4. **Historical coverage:** delivery-derived features depend on the match records present in `deliveries.csv`. The files should be checked for consistent coverage before making predictions for a period.
5. **No strong predictive claim:** none of the models beats the 53.0% majority baseline in accuracy. The results should not be represented as a reliable betting or forecasting system.
6. **Scope of prediction:** the current system uses information available after the toss and before the first ball. A system intended to predict before the toss would need to remove the toss-dependent inputs and be reevaluated.

### Useful questions to prepare for

- Why were two files needed instead of only `matches.csv`?
- What is the difference between a feature and the target variable?
- Why are historical team names standardised?
- What is data leakage, and how is it avoided here?
- Why was the data split chronologically instead of randomly?
- What does each of the six metrics tell us that accuracy alone does not?
- Why does Decision Tree rank first overall even though its accuracy is below the baseline?
- Why should the validation balanced accuracy not be reported as the final model performance?

