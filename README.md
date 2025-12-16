# SMS Spam Detection Using Machine Learning

An intelligent SMS spam detection system that uses machine learning to classify messages as spam or legitimate (ham). The system combines TF-IDF text vectorization, dimensionality reduction, handcrafted linguistic features, and XGBoost classification to achieve high accuracy.

## 🎯 Project Overview

This project addresses the growing problem of SMS spam by building an automated, data-driven detection system. Traditional rule-based filters struggle with evolving spam patterns, so we use machine learning to learn distinguishing patterns automatically.

### Key Features

- **High Accuracy**: Achieves 98.5% accuracy on test data
- **Low False Positives**: Maintains precision of 97.8%
- **Robust Performance**: AUC score of 99.1%
- **Real-time Classification**: Lightweight model suitable for production deployment
- **Comprehensive Feature Engineering**: Combines text and statistical features

## 👥 Team Members

- Saif Mohamed
- Muhab Abdelraouf
- Adham Tamer
- Ahmed Sharif
- Ahmed Ramadan
- Amr Hamoda

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.5% |
| Precision | 97.8% |
| Recall | 90.6% |
| AUC | 99.1% |

**Confusion Matrix:**
```
[[963   3]
 [ 14 135]]
```

## 🛠️ Technology Stack

- **Python 3.x**
- **scikit-learn**: Feature extraction and preprocessing
- **XGBoost**: Gradient boosting classifier
- **NLTK**: Text tokenization and stopword removal
- **Pandas & NumPy**: Data manipulation
- **TQDM**: Progress tracking

## 📁 Project Structure

```
sms-spam-detection/
├── data/
│   └── spam.csv                 # Dataset (not included, see below)
├── notebooks/
│   └── sms_spam_detection.ipynb # Full implementation notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Text cleaning and feature engineering
│   ├── model.py                 # Model training and prediction
│   └── utils.py                 # Helper functions
├── models/
│   └── trained_model.pkl        # Saved model (generated after training)
├── requirements.txt             # Project dependencies
├── README.md                    # This file
└── LICENSE                      # Project license
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). 

**Download the dataset:**
1. Download `spam.csv` from the link above
2. Place it in the `data/` directory

### Usage

#### Training the Model

```python
from src.model import SMSSpamDetector

# Initialize and train
detector = SMSSpamDetector()
detector.load_data('data/spam.csv')
detector.train()

# Evaluate
detector.evaluate()
```

#### Making Predictions

```python
# Predict single message
message = "Congratulations! You've won a free iPhone. Click here to claim."
result = detector.predict(message)
print(f"Prediction: {result['label']}")
print(f"Spam Probability: {result['probability']:.4f}")
```

#### Using the Notebook

Open and run `notebooks/sms_spam_detection.ipynb` for the complete implementation with visualizations and detailed explanations.

## 🔬 Methodology

### 1. Data Preprocessing
- Text cleaning (URL removal, special characters, lowercase conversion)
- Tokenization and stopword removal

### 2. Feature Engineering

**Text Features:**
- TF-IDF vectorization (top 3,000 features)
- Dimensionality reduction using Truncated SVD (200 components)

**Handcrafted Features:**
- Message length (character count)
- Number of words
- Number of sentences
- Average word length
- Stopword ratio

### 3. Model Training
- **Algorithm**: XGBoost Classifier
- **Hyperparameters**:
  - n_estimators: 300
  - max_depth: 8
  - learning_rate: 0.15
- **Custom threshold**: 0.3 for spam classification (optimized for recall)

### 4. Evaluation
Multiple metrics ensure comprehensive performance assessment:
- Accuracy, Precision, Recall
- ROC-AUC score
- Confusion matrix analysis

## 🌐 Real-World Applications

This system can be integrated into:
- Mobile network operator messaging pipelines
- Smartphone messaging applications
- Email filtering systems
- Customer support platforms

The lightweight architecture enables real-time processing with minimal latency.

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Multilingual spam detection
- [ ] Active learning for continuous improvement
- [ ] Web API for easy integration
- [ ] Mobile app demonstration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue in this repository.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI Machine Learning Repository
- scikit-learn and XGBoost communities
- NLTK project for natural language processing tools
