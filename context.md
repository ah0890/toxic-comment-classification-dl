# Project Context

Running record of the project's state, decisions, and findings. Updated at the end of every step.
Read this first when resuming work.

_Last updated: 2026-10-10 (Step 3 completed)_

## Project

- **Title:** Toxic Comment Classification Using Deep Learning
- **Course:** AIP301 hands-on assignment
- **Repository:** https://github.com/ah0890/toxic-comment-classification-dl (branch `main`)
- **Goal:** classify comments into toxicity categories with an LSTM (Embedding → LSTM → Dropout → Dense → Softmax), organised under the five assignment tasks in one notebook.

## Workflow rules

- Build strictly **one roadmap step at a time**; wait for the user's `DONE` before the next step.
- The user does all Git commits and pushes. The assistant never runs `git commit`, `git push`, or `git init`, and never changes the remote.
- Give explicit `git add <files>` commands (no `git add .`).
- Never invent results or metrics; every number in the notebook comes from actual execution.

## Roadmap progress

| Step | Title | Status |
|---|---|---|
| 1 | Repository inspection and project setup | ✅ Done (commit `2257cfc`) |
| 2 | Dataset loading and initial exploration | ✅ Done |
| 3 | Text cleaning and target analysis | ✅ Done (awaiting commit) |
| 4 | Data splitting and NLP tokenization | ⏳ Next |
| 5 | LSTM model development | Pending |
| 6 | Model training and validation | Pending |
| 7 | Model evaluation and visualization | Pending |
| 8 | Prediction on new comments (`src/predict.py`) | Pending |
| 9 | Documentation and final verification | Pending |

## Environment

- Windows 11. Default `python` is 3.14, which TensorFlow does not support.
- Python 3.12 is registered with the `py` launcher, but its folder has been deleted, so it does not work.
- **Project venv:** `.venv/`, created with `py -3.13` (ignored by Git).
- Installed: TensorFlow 2.21.0, tf_keras 2.21.0, Keras 3.15.1, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, nbconvert 7.17.1.
- Keras 3 removed the classic `Tokenizer`, so use `from tf_keras.preprocessing.text import Tokenizer` (verified to import).
- Run the notebook headless: `.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/toxic_comment_classification.ipynb`
- Run the tests: `.venv/Scripts/python.exe -m unittest discover tests`

## Project layout

| Path | Purpose |
|---|---|
| `notebooks/toxic_comment_classification.ipynb` | Main notebook (Task 1 and Task 2 done) |
| `src/preprocessing.py` | `clean_text()`, shared by the notebook and the future `src/predict.py` |
| `tests/test_preprocessing.py` | unittest tests for `clean_text()` |
| `data/toxic_comments_dataset.csv` | Dataset (ignored by Git) |
| `models/` | Saved model and artifacts (ignored by Git) |
| `outputs/plots/`, `outputs/reports/` | Saved charts and reports (committed) |

## Dataset facts (verified by execution)

- 30,000 rows × 15 columns, with 0 missing values and 0 duplicates (rows, IDs, text; also after cleaning).
- Columns: `Comment_ID`, `Comment_Text`, `Toxicity_Label`, `Clean`, `Toxic`, `Severe_Toxic`, `Threat`, `Identity_Attack`, `Toxicity_Score`, `Comment_Length`, `Word_Count`, `Platform`, `Topic`, `Moderation_Priority`, `Language` (only English).
- **Input column:** `Comment_Text` (about 250 chars and about 37 words each).
- **Target column:** `Toxicity_Label`, with 5 mutually exclusive classes:
  - Clean: 18,579 (61.93%)
  - Toxic: 6,693 (22.31%)
  - Severe_Toxic: 1,801 (6.00%)
  - Identity_Attack: 1,468 (4.89%)
  - Threat: 1,459 (4.86%)
  - The largest/smallest imbalance ratio is 12.7.
- The binary flags are multi-label (`Toxic`=1 for every toxic subtype; 4,728 rows have more than one flag). The rule "Clean → Severe_Toxic → Threat → Identity_Attack → else Toxic" reproduces `Toxicity_Label` for 100% of rows.
- **Template / near-duplicate finding:** every text is `<intro phrase> Source wording: Comment for review: "<core comment>". <closing phrase>`. There are only **17 distinct core comments** (4 Clean, 4 Toxic, 3 each for the other classes), and each always has the same label. `Comment_Length` equals the core comment's length.
  - Consequence: a random split puts the same core comments in both train and test, so expect very high test accuracy that does not prove generalisation.
  - Plan: state this limitation in the evaluation (Step 7) and test on genuinely new comments (Steps 7 and 8). An optional extra test that holds out whole core comments is an idea, not yet agreed.

## Decisions

- Target = `Toxicity_Label` (multi-class, Softmax). The binary columns are not used as targets.
- Cleaning: lowercase, decode HTML entities and remove tags, remove URLs and e-mails, drop apostrophes (`don't` → `dont`), replace non-letters (digits, punctuation, emojis) with a space, collapse whitespace. Stop words are kept, and the template wording is not stripped.
- Duplicates: exact duplicates with consistent labels are removed (0 found); conflicting duplicates are reported to `outputs/reports/conflicting_label_duplicates.csv` and kept for review (0 found). Near-duplicates (shared core comments) are kept.
- Planned encoding (Step 4): `LabelEncoder` (alphabetical) → integer labels → `sparse_categorical_crossentropy`, with an output of `Dense(5, softmax)`. Encode after the split and save the encoder for prediction.
- Split plan: 80/20, stratified on `Toxicity_Label`, with a fixed `random_state`. Fit the Tokenizer on training text only.

## Generated outputs

- `outputs/plots/target_distribution.png`: class distribution bar chart
- `outputs/reports/target_distribution.csv`: counts and percentages per class
- `outputs/reports/core_comment_summary.csv`: the 17 core comments with label and record count

## Notebook variables available after Task 2

- `df`: full cleaned DataFrame, with added columns `clean_text` and `core_comment`
- `model_df`: `Comment_ID`, `clean_text`, `Toxicity_Label`, shape (30000, 3)
- `PROJECT_ROOT`, `DATA_PATH`, `PLOTS_DIR`, `REPORTS_DIR`
