# NutriBench - Carbohydrate Prediction from Meal Descriptions

## Overview

**NutriBench** is a nutrition prediction task focused on estimating the **carbohydrate (carb) content** of meals from **free-text descriptions**. This project is part of the final assignment for **ECE 180: Deep Learning Applications** at UCSB (Spring 2025).

Given a dataset of meal descriptions and their corresponding carbohydrate values, our team is tasked with:

- Designing and implementing at least three model variations (using different feature extraction and architectures).
- Comparing performance across models using validation metrics such as MAE.
- Selecting the best-performing model to generate predictions on a test set.
- Compiling a full pipeline with reproducible code and a well-documented report.

We utilize **PyTorch** as the core framework, following best practices in feature extraction (e.g., TF-IDF, SBERT embeddings), machine learning and deep learning modeling (e.g., Ridge Regression, MLP, XGBoost, Transformers), and evaluation.

---

## Workflow

This project follows a modular workflow structured into the following key stages:

### 1. Data Preprocessing

- **Method 1: TF-IDF Vectorization**
  - Uses `TfidfVectorizer` to extract sparse features from text.
- **Method 2: Sentence-BERT Embeddings**
  - Uses the `"all-MiniLM-L6-v2"` pre-trained model to create dense sentence embeddings.
- **Method 3: BERT Tokenizer**
  - Uses the `"BERT Tokenizer"` to extract features from text.
Processed data for each method is saved under:

- `data/method1_tfidf/`
- `data/method2_sbert/`

---

### 2. Model Training & Validation

Each model is developed as a separate module under `src/`, and trained on the corresponding feature representation.

| Model           | Description                            | Folder                  |
|------------------|----------------------------------------|--------------------------|
| **TF-IDF → Ridge** | L2-regularised linear regression | `src/ridge_model/` |
| **TF-IDF → MLP** | 3-layer feed-forward network | `src/mlp_model/` |
| **TF-IDF → XGBoost** | Bayesian-tuned gradient-boosted trees | `src/xgboost_model/` |
| **BERT → MLP** | Fine-tuned `bert-base-uncased` + 2-layer head | `src/bert_model/` |
| **SBERT → Transformer** | Frozen MiniLM embeddings + 6-layer transformer regressor & hybrid loss | `src/transformer_model/` |

Each model outputs predictions, metrics, and plots under the corresponding subfolder in `output/`:

- `output/ridge/`
- `output/mlp/`
- `output/xgboost/`
- `output/bert/`
- `output/transformer/`

---

### 3. Evaluation

- Models are evaluated using **Mean Absolute Error (MAE)**.
- We conduct **error analysis** and visualize predicted vs. true carb values.
- All insights are included in the final report and used to select the best model for test predictions.

---

### 4. Final Output

- `test_predictions.csv` with predicted carb values
- `report/report.pdf` containing:
  - Introduction
  - Methods (data processing, model design, training strategy)
  - Validation results
  - Model comparison
  - Conclusion
- All organized code under `src/` and `output/`

---

## Repository Structure

```
.
├── README.md                  # Project overview (this file)
├── requirements.txt           # Python dependencies
├── report/                    # Final report and figures
│ ├── method1_tfidf/           # Processed data using TF-IDF
│ └── method2_sbert/           # Processed data using Sentence-BERT
├── src/                       # Source code for each model
│ ├── mlp_model/               # MLP implementation on different feature sets
│ ├── ridge_model/             # Linear regression based model
| ├── xgboost_model/           # Gradient‑boosted decision‑tree model
| ├── bert_model/              # Fine-tuned BERT encoder + 2-layer MLP regression head
│ └── transformer_model/       # Transformer-based regression model (e.g., fine-tuned BERT)
└── output/                    # Model predictions and result logs
├── mlp/                       # Output CSVs, plots, logs from MLP
├── ridge/                     # Output CSVs, plots, logs from Ridge
├── xgboost/                   # Output CSVs, plots, logs from XGBoost
├── bert/                      # Output CSVs, plots, logs from Bert
└── transformer/               # Output CSVs, plots, logs from Transformer
```
