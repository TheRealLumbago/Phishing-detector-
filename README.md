# Phishing Email Detection System

A comprehensive machine learning project for detecting phishing emails using multiple classification algorithms, deep learning models, and interpretability techniques.

## 📋 Project Overview

This project implements and compares various machine learning and deep learning approaches for phishing email detection, including traditional ML models (Logistic Regression, Naive Bayes, Random Forest) and advanced deep learning models (Bidirectional LSTM). The project also includes model interpretability analysis using SHAP, LIME, and ELI5, along with hyperparameter optimization.

## ✨ Features

### Machine Learning Models
- **Logistic Regression** - Linear classifier with optimized hyperparameters
- **Naive Bayes** - Probabilistic classifier
- **Random Forest** - Ensemble method with hyperparameter tuning
- **Bidirectional LSTM** - Deep learning model for sequence classification

### Advanced Features
- **Model Interpretability**
  - SHAP (SHapley Additive exPlanations) for deep learning models
  - LIME (Local Interpretable Model-agnostic Explanations) for text explanations
  - ELI5 for feature importance visualization
  - Feature importance analysis for traditional models

- **Hyperparameter Optimization**
  - RandomizedSearchCV for Random Forest and Logistic Regression
  - Keras Tuner for LSTM model optimization
  - Learning rate scheduling and early stopping

- **Agent-Based Approach**
  - Graph-based BFS search for phishing pattern detection
  - Rule-based feature extraction and classification

### Evaluation Metrics
- Accuracy
- F1-Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix
- Learning Curves

## 🛠️ Technologies Used

- **Python 3.10.6**
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow 2.20.0, Keras
- **Interpretability**: SHAP, LIME, ELI5
- **Optimization**: Keras Tuner, scikit-learn RandomizedSearchCV
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Text Processing**: TF-IDF Vectorization, Tokenization

## 📦 Installation

### Prerequisites
- Python 3.10.6 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/phishing-email-detection.git
cd phishing-email-detection
```

### Step 2: Install Required Packages
```bash
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow shap lime eli5 keras-tuner scipy
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

### Step 3: GPU Support (Optional)
For GPU acceleration with TensorFlow:
1. Install CUDA Toolkit 12.x from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Install cuDNN 8.9 for CUDA 12.x from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
3. Restart your computer
4. Verify GPU detection:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 📁 Project Structure

```
phishing-email-detection/
│
├── main.ipynb                 # Main notebook with all ML/DL models and analysis
├── simpleagent.py             # Agent-based phishing detection implementation
├── Phishing_Email.csv         # Dataset (phishing and safe emails)
├── README.md                  # Project documentation
│
└── untitled_project/          # Keras Tuner results (can be regenerated)
    └── trial_*/               # Hyperparameter tuning trial results
```

## 🚀 Usage

### Running the Main Notebook

1. Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code
2. Ensure you're using Python 3.10.6 kernel
3. Run cells sequentially:
   - **Cell 0**: Import libraries and check GPU availability
   - **Cell 1-5**: Load and preprocess data
   - **Cell 6-8**: Train traditional ML models (Logistic Regression, Naive Bayes, Random Forest)
   - **Cell 9**: Confusion matrix visualization
   - **Cell 10**: Build and train LSTM model
   - **Cell 11**: Model interpretability analysis (SHAP, LIME, ELI5)
   - **Cell 12**: Hyperparameter optimization (Random Forest, Logistic Regression, LSTM)
   - **Cell 13**: Basic model comparison
   - **Cell 14**: Voting Ensemble
   - **Cell 15**: ROC curves visualization
   - **Cell 16**: Stacking Ensemble
   - **Cell 17**: Comprehensive model comparison with all metrics
   - **Cell 18**: Final results summary

### Running the Agent-Based Approach

```python
from simpleagent import PhishingAgent

agent = PhishingAgent()

# Test with sample emails
phishing_email = "Your account has been suspended. Please click here to verify..."
safe_email = "Hello team, Just a reminder about the meeting scheduled for tomorrow."

print(agent.classify(phishing_email))  # Output: Phishing Email
print(agent.classify(safe_email))      # Output: Safe Email
```

## 📊 Results

The project compares multiple models and provides comprehensive analysis:

### Performance Metrics
- **Model Performance Comparison**: Accuracy, Precision, Recall, F1-Score, ROC-AUC metrics
- **Comprehensive Comparison Table**: Side-by-side comparison of all models
- **Model Ranking**: Performance ranking by ROC-AUC score

### Visualizations
- **Confusion Matrices**: Visual representation of model predictions
- **ROC Curves**: Comparison of model performance across different thresholds with AUC scores
- **Learning Curves**: Training and validation performance over epochs for LSTM
- **Feature Importance**: Top contributing features for each model (Random Forest, Logistic Regression)
- **Multi-metric Comparison Charts**: Bar charts comparing all metrics across models

### Interpretability
- **SHAP Explanations**: Deep learning model explanations (GradientExplainer)
- **LIME Explanations**: Local explanations for individual email predictions with text highlighting
- **ELI5 Explanations**: Feature importance visualization for linear models

### Ensemble Methods
- **Voting Ensemble**: Soft voting classifier combining Logistic Regression, Random Forest, and Naive Bayes
- **Stacking Ensemble**: Meta-learner approach with cross-validation

## 🔍 Model Interpretability

### SHAP Values
- Deep learning model explanations using GradientExplainer
- Feature contribution analysis for LSTM predictions

### LIME Explanations
- Local explanations for individual email predictions
- Text-based feature importance highlighting

### Feature Importance
- Random Forest feature importance rankings
- Logistic Regression coefficient analysis
- ELI5 explanations for linear models

## 🎯 Key Findings

### Model Performance Results

Based on comprehensive evaluation on the test set:

- **Best Performing Model**: Logistic Regression / LSTM (varies by metric)
- **Top Accuracy**: ~96.2% (Logistic Regression, Random Forest, LSTM)
- **Top F1-Score**: ~0.96 (across multiple models)
- **Top ROC-AUC**: >0.98 (all models show excellent discrimination)

### Detailed Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 96.22% | 0.96 | 0.96 | 0.96 | >0.98 |
| Naive Bayes | 94.29% | 0.94 | 0.94 | 0.94 | >0.97 |
| Random Forest | 96.03% | 0.96 | 0.96 | 0.96 | >0.98 |
| Bidirectional LSTM | ~96%+ | ~0.96 | ~0.97 | ~0.96 | >0.98 |
| Voting Ensemble | 96.49% | 0.96 | 0.96 | 0.96 | >0.98 |
| Stacking Ensemble | ~96%+ | ~0.96 | ~0.96 | ~0.96 | >0.98 |

### Key Insights

1. **All models achieved excellent performance** (>94% accuracy), demonstrating the effectiveness of different approaches
2. **Traditional ML models** (Logistic Regression, Random Forest) perform comparably to deep learning
3. **Ensemble methods** (Voting & Stacking) provide robust and consistent predictions
4. **LSTM model** shows competitive performance with sequence-based learning
5. **Model interpretability** tools successfully explain predictions, enhancing trust and understanding
6. **Hyperparameter optimization** improved model performance across all algorithms

## 📝 Dataset

The dataset (`Phishing_Email.csv`) contains:
- Email text content
- Email type labels (Phishing Email / Safe Email)
- Preprocessed and ready for analysis

## 🔧 Hyperparameter Optimization

### Random Forest
- Optimized: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

### Logistic Regression
- Optimized: `C`, `penalty`, `solver`, `max_iter`

### LSTM
- Optimized: `embedding_dim`, `lstm_units`, `dropout`, `dense_units`, `optimizer`

## 📈 Progress Report III Features

This project implements Progress Report III requirements:
- ✅ Advanced ML/DL model (Bidirectional LSTM)
- ✅ Model interpretability (SHAP, LIME, ELI5)
- ✅ Hyperparameter optimization (RandomizedSearchCV, Keras Tuner)
- ✅ Comprehensive model comparison and evaluation
- ✅ ROC curves and AUC analysis
- ✅ Ensemble methods (Voting & Stacking)
- ✅ Complete performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- ✅ Final results summary and key findings

## 📄 Files

- `main.ipynb` - Main Jupyter notebook with all implementations and analysis
- `simpleagent.py` - Agent-based phishing detection with BFS graph search
- `Phishing_Email.csv` - Dataset containing phishing and safe emails
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation


