# Building Damage Assessment Using Deep Learning

## Project Overview

This project focuses on binary classification of buildings using satellite images. It identifies whether a building is damaged or not damaged.

The project uses deep learning models to classify images into two categories:

- `damage`
- `no_damage`

The dataset contains satellite images of buildings from the Greater Houston area following Hurricane Harvey.

## Models Implemented

Three model architectures are explored:

1. **Convolutional Neural Network (CNN)**
2. **EfficientNetB0**
3. **SegFormer-inspired classifier**

The models are trained and evaluated separately, and their performance is compared using classification metrics.

## Evaluation Metrics

The evaluation includes:

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Confusion matrix
- Precision-Recall analysis
- Threshold sensitivity analysis

Additional analysis includes duplicate-image checks, geographic analysis, and model explainability experiments.

## Explainable AI

The Flask application provides:

- Prediction confidence
- Damage and no-damage probabilities
- Saliency maps
- Grad-CAM visualizations

These visualizations help interpret which image regions influence model predictions.

## Flask Web Application

The application includes:

- User registration and login
- Password hashing
- SQLite database
- Protected prediction page
- Image upload
- Damage classification
- Prediction probabilities
- Saliency map and Grad-CAM visualization

## Project Structure

```text
building_damage_flask/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- training/
|-- models/
|-- results/
|-- final_results/
|-- claude_paper_evidence/
|
|-- static/
|-- templates/
|
|-- analyze_*.py
|-- evaluate_*.py
|-- train_*.py
```

## Dataset

The dataset is not included in this repository.

Place the cleaned dataset in the following local directory:

```text
data/clean_dataset/
```

The dataset should contain the two class folders:

```text
clean_dataset/
|-- damage/
|-- no_damage/
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd building_damage_flask
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Model Weights

Trained model weights are not included in this repository because of their file sizes.

The Flask application currently expects the evaluated CNN model at:

```text
results/CNN_Clean_Training/best_cnn_clean.keras
```

Obtain the trained model separately and place it at that exact path before running the application.

## Run the Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## Security Notes

- The SQLite database is generated locally.
- Dataset files and trained model weights are excluded from Git.
- Set a secure `FLASK_SECRET_KEY` environment variable before deployment.
- Do not commit passwords, secret keys, or private user data.

## Technologies Used

- Python
- TensorFlow / Keras
- PyTorch
- Flask
- NumPy
- Matplotlib
- Pillow
- SQLite
- Scikit-learn

## Project Purpose

This project demonstrates the application of deep learning to post-disaster building damage assessment using satellite imagery, with a web-based interface for image classification and visual explanation.
