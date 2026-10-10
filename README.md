# Toxic Comment Classification Using Deep Learning

Deep learning-based toxic comment classification using Python, TensorFlow/Keras, and NLP. Includes data exploration, text cleaning, tokenization, sequence padding, LSTM model training, and evaluation using accuracy, loss, confusion matrix, classification report, and performance visualizations.

> Academic project for the **AIP301** hands-on assignment.
> Status: **in development** (project setup complete; the notebook and model are added step by step).

## Objectives

The project covers the five assignment tasks:

1. **Dataset Loading and Initial Exploration** – load the CSV with Pandas and inspect its records, shape, columns, data types, statistics, missing values, and duplicates.
2. **Text Cleaning, Preprocessing, Data Quality, and Target Analysis** – clean comment text (lowercase, remove HTML, URLs, punctuation, digits, emojis, extra spaces), handle missing values and duplicates, and analyse the target-label distribution.
3. **NLP Tokenization and Sequence Padding** – fit a Keras Tokenizer on the training text only, convert text to integer sequences, and pad/truncate them to a fixed length.
4. **Deep Learning Model Development, Training, and Validation** – build and train an LSTM model (Embedding → LSTM → Dropout → Dense → Softmax) with a validation split.
5. **Model Evaluation, Visualization, and Prediction** – report test accuracy and loss, a classification report, a confusion matrix, training curves, and predictions on new comments.

## Tech Stack

- Python 3.10–3.13
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Project Structure

```
toxic-comment-classification-dl/
├── data/                 # Place toxic_comments_dataset.csv here (not committed to Git)
├── notebooks/            # Jupyter notebook organised by the five assignment tasks
├── src/                  # Reusable Python code (e.g. prediction script)
├── models/               # Saved model and tokenizer (not committed to Git)
├── outputs/
│   ├── plots/            # Saved charts (accuracy/loss curves, confusion matrix, ...)
│   └── reports/          # Saved evaluation reports
├── requirements.txt
└── README.md
```

## Dataset

Place the dataset file at:

```
data/toxic_comments_dataset.csv
```

The dataset is excluded from Git by `.gitignore`. The text and target columns will be identified from the actual CSV schema during the exploration step.

## Setup (Windows)

TensorFlow requires Python 3.10–3.13, so create the virtual environment with Python 3.13:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Detailed instructions for running the notebook, training, evaluation, and prediction will be added as the project is completed.
