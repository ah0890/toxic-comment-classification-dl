# Project Context

Running record of the project's state, decisions, and findings. Updated at the end of every step.
Read this first when resuming work.

_Last updated: 2026-10-10 (Step 5 completed)_

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
| 2 | Dataset loading and initial exploration | ✅ Done (commit `ba1ac74`) |
| 3 | Text cleaning and target analysis | ✅ Done (commit `abbad01`) |
| 4 | Data splitting and NLP tokenization | ✅ Done (commit `d1ced45`) |
| 5 | LSTM model development | ✅ Done (awaiting commit) |
| 6 | Model training and validation | ⏳ Next |
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
| `notebooks/toxic_comment_classification.ipynb` | Main notebook (Tasks 1–3 done, Task 4 model-building part done). Notebook Task 3 = roadmap Step 4; Task 4 = Steps 5+6 |
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
- Split: 80/20, `stratify=Toxicity_Label`, `random_state=42`, giving 24,000 train and 6,000 test. The class percentages match within 0.02 points.
- Label encoding: `LabelEncoder` fitted on the training labels. Mapping: Clean=0, Identity_Attack=1, Severe_Toxic=2, Threat=3, Toxic=4. Integer labels are used with `sparse_categorical_crossentropy` and an output of `Dense(5, softmax)`.
- Tokenizer: `tf_keras` `Tokenizer(num_words=10000, oov_token="<OOV>")`, fitted on training text only. The real vocabulary is just **211 tokens** (incl. `<OOV>`), so `VOCAB_SIZE = 212` (incl. padding index 0). The test set has 0.00% OOV tokens (templated data).
- Sequences: training lengths run from 30 to 46 tokens (median 37). `MAX_LEN = 50` (the max rounded up to a multiple of 10), so 100% fit without truncation. Padding `pre`, truncating `post`.
- Model (Step 5): Keras 3 (`tensorflow.keras`) `Sequential` named `toxic_comment_lstm`: Input(50) → Embedding(212, 64) → LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(5, softmax). **48,837 params**, which match the formulas. Compiled with Adam(lr=0.001), `sparse_categorical_crossentropy` and `accuracy`. No `mask_zero`, since pre-padding keeps the real words last. Seeded with `keras.utils.set_random_seed(42)`.
- Training runs on the CPU (TensorFlow has no GPU support on native Windows).
- Tokenizer, label encoder and settings (MAX_LEN, padding, truncating) are **not saved yet**. Save them in Step 6 with the model, for `predict.py`.

## Generated outputs

- `outputs/plots/target_distribution.png`: class distribution bar chart
- `outputs/reports/target_distribution.csv`: counts and percentages per class
- `outputs/reports/core_comment_summary.csv`: the 17 core comments with label and record count
- `outputs/plots/sequence_length_distribution.png`: histogram of training sequence lengths with a MAX_LEN line
- `outputs/reports/model_summary.txt`: Keras model summary

## Notebook variables available after Task 4 (model building)

- `df`: full cleaned DataFrame, with added columns `clean_text` and `core_comment`
- `model_df`: `Comment_ID`, `clean_text`, `Toxicity_Label`, shape (30000, 3)
- `PROJECT_ROOT`, `DATA_PATH`, `PLOTS_DIR`, `REPORTS_DIR`
- `RANDOM_STATE` (42); seeds set for `random`, NumPy and TensorFlow
- `X_train_text`, `X_test_text`, `y_train_labels`, `y_test_labels`: split text and label Series
- `X_train` (24000, 50) and `X_test` (6000, 50): int32 padded sequences
- `y_train` (24000,) and `y_test` (6000,): integer labels
- `label_encoder`, `CLASS_NAMES`, `NUM_CLASSES` (5)
- `tokenizer`, `VOCAB_SIZE` (212), `MAX_WORDS`, `OOV_TOKEN`, `MAX_LEN` (50), `PADDING`, `TRUNCATING`
- `keras`, `layers`, `model` (compiled, untrained), `EMBEDDING_DIM`, `LSTM_UNITS`, `DENSE_UNITS`, `DROPOUT_RATE`, `LEARNING_RATE`
