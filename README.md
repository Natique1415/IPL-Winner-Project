# DMDW Group Project ( IPL )

Use the classification techniques taught in class to predict IPL match winners. See `installation.md` for setup.

## Step 1: Data Collection
**Location:** Dataset files in project root
- Selected the [Kaggle IPL Complete Dataset by `patrickb1912`](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020/).
- The URL still says 2008-2020, but the version used covers seasons 2008 to 2024 (1,095 matches in total).
- Downloaded `matches.csv` and `deliveries.csv`.

## Step 2: Data Understanding and Exploration
**File:** `Step-2/viewing-data.py`
- Loaded both CSV files (`matches.csv`, `deliveries.csv`) using Pandas.
- Inspected dataset dimensions, column names and missing values.
- Identified columns with missing values and assessed their significance.
- Counted matches per team name and identified potential historical naming variations (`Step-2/team-renaming.py`).


## Step 3: Data Preprocessing
**File:** `Step-3/cleaning-data.py`
- Checked for duplicate records: none found in either dataset.
- Created a classification-specific dataset with 1,090 matches by excluding 5 matches with unknown winners.
- Retained all 260,920 delivery records.
- Standardized historical franchise names in working copies using a mapping dictionary:
	- Delhi Daredevils, Delhi Capitals → Canonical name: Delhi Capitals
	- Kings XI Punjab, Punjab Kings → Canonical name: Punjab Kings
	- Royal Challengers Bangalore, Royal Challengers Bengaluru → Canonical name: Royal Challengers Bengaluru
	- Rising Pune Supergiant, Rising Pune Supergiants → Canonical name: Rising Pune Supergiant
- Kept distinct franchises separate, including Deccan Chargers, Sunrisers Hyderabad, Kochi Tuskers Kerala, Gujarat Lions and Gujarat Titans.
- Preserved the original CSV files unchanged (all changes were made on `.copy()` versions).


## Step 4: Classification

**Folder:** `Step-4/Classification/`

**Files:**
- `classification-2.py` : Initial model with historical performance features.
- `classification-3.py` : Improved model with feature reduction and hyperparameter tuning.

### Objective
Predict whether Team 1 or Team 2 wins an IPL match using a Decision Tree Classifier.

### Algorithm
- Decision Tree using entropy and Information Gain.
- `max_depth`, `min_samples_leaf`, and `class_weight` selected using chronological validation.
- Final evaluation performed on a separate chronological test set.

### Features Used
- `team1_won_toss` : Whether Team 1 won the toss.
- `team1_bats_first` : Whether Team 1 bats first.
- `prior_win_rate_diff` : Difference between the teams' historical win rates.
- `recent_win_rate_diff` : Difference between their win rates over their previous five matches.

Predictions are assumed to be made after the toss and before the first ball, so toss information is allowed. Post-match columns (result margin, player of the match, final scores) are excluded to avoid data leakage.

### Data Preparation and Evaluation
- Standardized historical franchise names.
- Removed duplicate matches and matches with invalid or unknown winners.
- Created the target variable `team1_won` (1 = Team 1 wins, 0 = Team 2 wins).
- Used chronological training, validation, and testing periods to avoid randomly mixing past and future matches.
- Evaluated the model using accuracy, precision, recall, specificity, F1-score, balanced accuracy, and a confusion matrix.
- Generated `decision_tree_classification3.png` and `classification_metrics_classification3.png`.

### Classification-3.py Results
- Usable matches: 1,090.
- Training and validation pool: 872 matches (714 training + 158 validation).
- Testing matches: 218.
- Test accuracy: 48.6%.
- Balanced accuracy: 49.1%.
- Precision: 50.0%.
- Recall: 32.1%.
- Specificity: 66.0%.
- F1-score: 39.1%.
- Majority-class baseline accuracy: 51.4% (accuracy from always predicting the more common outcome).

**Current status:** Classification 3 improves recall and F1-score substantially compared with Classification 2, but its test accuracy remains below the majority-class baseline. Further improvement is required.

---

### Mistakes in the Initial Approach and Improvements in Classification 3

**1. Too many categorical features**

- **Earlier approach:** Included venue and toss decision as categorical predictors, creating many encoded features that could encourage unhelpful splits.
- **Change in Classification 3:** Removed these categorical features and used four numerical features describing toss outcome, batting order, and historical team performance.

**2. Predictions heavily favored Team 2**

- **Earlier approach:** Classification 2 predicted Team 1 wins for only 5 of 218 test matches, producing just 1.8% recall.
- **Change in Classification 3:** Simplified the feature set and tuned the tree using validation balanced accuracy. Team 1 recall improved to 32.1%.

**3. Fixed hyperparameters without systematic selection**

- **Earlier approach:** Used predefined tree settings without comparing alternatives through a separate validation period.
- **Change in Classification 3:** Tested different tree depths, minimum leaf sizes, and class-weight settings, selecting the best combination using validation balanced accuracy.

**4. Evaluation focused too heavily on accuracy**

- **Earlier approach:** Accuracy alone did not reveal that the model almost never predicted Team 1 wins.
- **Change in Classification 3:** Added precision, recall, specificity, F1-score, balanced accuracy, and a confusion matrix for a more complete evaluation.

**5. Model performance was not compared against a baseline**

- **Earlier approach:** The model's accuracy was evaluated without sufficient emphasis on a simple benchmark.
- **Change in Classification 3:** Compared test accuracy against a majority-class baseline of 51.4%.


## Step 5: Naive Bayes Classification
**Folder:** `Step-5/Naive-Bayes/`

- **Objective:** Predict whether Team 1 or Team 2 wins an IPL match.
- **Algorithm:** Categorical Naive Bayes, following the conditional-probability method taught in class.
- **Target:** `team1_won` (1 = Team 1 wins, 0 = Team 2 wins).
- **Attributes:** `team1_won_toss`, `team1_bats_first`, `prior_win_rate_diff`, and `recent_win_rate_diff`.
- Categorize numerical win-rate differences into Team 1 stronger, Similar strength, and Team 2 stronger.
- Estimate class priors and conditional probabilities using training data, with Laplace smoothing.
- Evaluate using accuracy, precision, recall, specificity, F1-score, balanced accuracy, and a confusion matrix.

### Results
- Test accuracy: 48.2%.
- Precision: 49.4%.
- Recall: 35.7%.
- Specificity: 61.3%.
- F1-score: 41.5%.
- Balanced accuracy: 48.5%.
- Saved evaluation metrics to `Step-7/Model-Comparison/naive_bayes_metrics.json`.
- Generated `classification_metrics_naive_bayes.png`.

**Status:** Completed. Accuracy remains below the 51.4% majority-class baseline.


## Step 6: K-Nearest Neighbours (KNN) Classification
**Folder:** `Step-6/KNN/`

- **Objective:** Predict whether Team 1 or Team 2 wins an IPL match.
- **Algorithm:** K-Nearest Neighbours (KNN) using Euclidean distance.
- **Target:** `team1_won` (1 = Team 1 wins, 0 = Team 2 wins).
- **Attributes:** `team1_won_toss`, `team1_bats_first`, `prior_win_rate_diff`, and `recent_win_rate_diff`.
- Apply `StandardScaler` to normalize feature scales before calculating distances.
- Select the value of K and voting strategy using chronological validation and balanced accuracy.
- Train the final model using the training and validation data.
- Evaluate using accuracy, precision, recall, specificity, F1-score, balanced accuracy, and a confusion matrix.

### Results
- Selected settings: `n_neighbors=5`, `weights=uniform`.
- Test accuracy: 46.8%.
- Precision: 47.8%.
- Recall: 38.4%.
- Specificity: 55.7%.
- F1-score: 42.6%.
- Balanced accuracy: 47.0%.
- Saved evaluation metrics to `Step-7/Model-Comparison/knn_metrics.json`.
- Generated `classification_metrics_knn.png`.

**Status:** Completed. Accuracy remains below the 51.4% majority-class baseline.


## Step 7: Classification Model Comparison

**Folder:** `Step-7/Model-Comparison/`

**Files:**
- `compare-models.py` : Loads the saved JSON metrics and compares the three classifiers.
- `decision_tree_metrics.json` : Decision Tree evaluation metrics.
- `naive_bayes_metrics.json` : Naive Bayes evaluation metrics.
- `knn_metrics.json` : KNN evaluation metrics.
- `classification_comparison.png` : Grouped bar chart comparing evaluation metrics.

### Objective
Compare Decision Tree, Naive Bayes, and KNN performance using the same evaluation metrics.

### Results

| Metric | Decision Tree | Naive Bayes | KNN |
|---|---:|---:|---:|
| Accuracy | 48.6% | 48.2% | 46.8% |
| Precision | 50.0% | 49.4% | 47.8% |
| Recall | 32.1% | 35.7% | 38.4% |
| Specificity | 66.0% | 61.3% | 55.7% |
| F1-score | 39.1% | 41.5% | 42.6% |
| Balanced accuracy | 49.1% | 48.5% | 47.0% |

- Decision Tree had the highest accuracy (48.6%), precision (50.0%), specificity (66.0%), and balanced accuracy (49.1%).
- KNN had the highest recall (38.4%) and F1-score (42.6%).
- Naive Bayes was in between on most metrics.
- All three models had test accuracy below the 51.4% majority-class baseline.

### Reading the Results Carefully
- The test set has 218 matches, so one match is about 0.46 percentage points.
- Decision Tree vs Naive Bayes is a gap of about 1 match. Decision Tree vs KNN is about 4 matches. The best model vs the baseline is about 6 matches.
- Differences this small are within noise, so the ranking of the three models should not be treated as conclusive.
- All three models use the same four features, so the comparison mostly measures how each algorithm handles the same limited signal.

### Conclusion
Decision Tree scored best on accuracy and balanced accuracy, while KNN scored best on recall and F1-score, but the gaps are too small on 218 test matches to name a clear winner. None of the classifiers beat the majority-class baseline in accuracy. The results suggest that the current features (toss, batting order, and win-rate history) carry limited predictive power for IPL match outcomes, and that better features, not a different algorithm, are the main route to improvement.
