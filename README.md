# FraudGuard

FraudGuard is a beginner-friendly Flask dashboard for exploring a transaction CSV, training a fraud classifier, and testing individual transactions. It never creates placeholder records or reports invented metrics. The included workspace did not contain `data/excel.csv`, so the dashboard explains how to configure a real dataset until one is added.

## Project layout

```text
FraudGuard/
├── app.py
├── fraudguard_ml.py
├── train_model.py
├── requirements.txt
├── data/
│   └── excel.csv              # Add your real dataset here
├── models/                    # Trained model is saved here
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## 1. Set up in the VS Code terminal

Open this project folder in VS Code, then open **Terminal → New Terminal**. From the project directory, create and activate a virtual environment:

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the VS Code terminal's Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter with **Python: Select Interpreter** if it is not selected automatically.

## 2. Add and configure your dataset

Place your actual, header-row CSV at `data/excel.csv`. FraudGuard reads the column names from that file; it does not assume or invent transaction fields.

The default target column is `is_fraudulent`. If your target has a different name, set it in the terminal before training and starting the app:

```powershell
$env:FRAUD_TARGET_COLUMN = "replace with the exact target column"
```

For target values such as `Fraud` / `Legitimate`, `fraudulent` / `legitimate`, or numeric `1` / `0`, FraudGuard identifies the fraud class. For other label values, also set the exact fraud value:

```powershell
$env:FRAUD_POSITIVE_LABEL = "replace with the exact fraud label"
```

If the CSV is elsewhere, set its path as well:

```powershell
$env:FRAUDGUARD_DATASET_PATH = "D:\path\to\your\transactions.csv"
```

The target column must have exactly two non-missing class values. Rows with missing target labels are excluded from model training and evaluation, never assigned a label; the training script reports how many rows were excluded and how many remain. The dashboard's class counts also exclude rows without target labels, while the dataset row count and feature charts still reflect the full CSV. Missing numeric feature values are median-imputed, and missing categorical values are filled with the most frequent training value. The target is excluded from all model inputs.

## 3. Train and run

Train the model from the project directory:

```powershell
python train_model.py
```

The training script makes a stratified 80/20 train/test split before fitting preprocessing. Numeric and categorical preprocessing is fitted only on the training data through a scikit-learn pipeline, helping prevent preprocessing leakage. It reports held-out accuracy, precision, recall and F1 score, then saves the pipeline and dataset fingerprint to `models/fraudguard_model.joblib`.

Start the local web app:

```powershell
python app.py
```

Open the local address printed in the terminal (usually `http://127.0.0.1:5000`). Keep the target and dataset environment variables set in the same terminal when training and running the app. If the CSV changes after training, train again; the app will not present an out-of-date model as current.

## Dashboard features

- Dataset row count and missing-value handling based on the CSV on disk.
- Fraud / legitimate class counts and fraud percentage only when a valid target is configured.
- A real fraud-versus-legitimate chart generated from the loaded target labels.
- Dynamic prediction fields based on the dataset's non-target columns.
- Held-out model metrics and a predicted class with model probability.
- Responsive dark dashboard layout; all transaction analysis runs locally.

The model is an educational baseline, not a financial decision system. Evaluate the dataset, class balance, and false-positive/false-negative costs before using predictions operationally.
