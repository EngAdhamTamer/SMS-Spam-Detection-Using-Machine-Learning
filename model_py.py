"""
SMS Spam Detection Model
Main class for training and predicting spam messages
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import pickle
from .preprocessing import clean_text, handcrafted_features
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SMSSpamDetector:
    """
    SMS Spam Detection System using XGBoost and TF-IDF features
    """
    
    def __init__(self, max_features=3000, n_svd_components=200, 
                 threshold=0.5, test_size=0.2, random_state=42):
        """
        Initialize the SMS Spam Detector
        
        Args:
            max_features: Maximum number of TF-IDF features
            n_svd_components: Number of components for SVD dimensionality reduction
            threshold: Probability threshold for spam classification
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.n_svd_components = n_svd_components
        self.threshold = threshold
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components
        self.tfidf = None
        self.svd = None
        self.scaler = None
        self.classifier = None
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath, encoding='latin-1'):
        """
        Load and prepare the dataset
        
        Args:
            filepath: Path to the CSV file
            encoding: File encoding (default: latin-1)
        """
        df = pd.read_csv(filepath, encoding=encoding)
        
        # Keep only necessary columns
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
        
        # Convert labels to binary
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        df = df.dropna().reset_index(drop=True)
        
        self.df = df
        print(f"Dataset loaded: {len(df)} messages")
        print(f"Spam: {sum(df['label'] == 1)}, Ham: {sum(df['label'] == 0)}")
        
    def preprocess_and_extract_features(self):
        """
        Preprocess text and extract features
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Cleaning text...")
        texts = self.df["text"].astype(str).apply(clean_text)
        y = self.df["label"].values
        
        # TF-IDF vectorization
        print("Extracting TF-IDF features...")
        self.tfidf = TfidfVectorizer(max_features=self.max_features, 
                                      stop_words="english")
        X_tfidf = self.tfidf.fit_transform(texts)
        
        # SVD dimensionality reduction
        print("Applying SVD dimensionality reduction...")
        n_svd = min(self.n_svd_components, X_tfidf.shape[1] - 1)
        if n_svd < 2:
            n_svd = 2
        self.svd = TruncatedSVD(n_components=n_svd, random_state=self.random_state)
        X_svd = self.svd.fit_transform(X_tfidf)
        
        # Handcrafted features
        print("Extracting handcrafted features...")
        X_hand = handcrafted_features(texts)
        
        # Combine features
        X = np.hstack([X_svd, X_hand])
        print(f"Final feature shape: {X.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, 
            random_state=self.random_state
        )
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train(self, n_estimators=300, max_depth=8, learning_rate=0.15):
        """
        Train the XGBoost classifier
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        if self.X_train is None:
            raise ValueError("Features not extracted. Call preprocess_and_extract_features() first.")
        
        print("Training XGBoost classifier...")
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=self.random_state
        )
        
        self.classifier.fit(self.X_train, self.y_train)
        print("Training complete!")
        
    def evaluate(self):
        """
        Evaluate the model on test data
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        prob = self.classifier.predict_proba(self.X_test)[:, 1]
        pred = (prob >= self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred)
        recall = recall_score(self.y_test, pred)
        auc = roc_auc_score(self.y_test, prob)
        cm = confusion_matrix(self.y_test, pred)
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(self.y_test, pred, 
                                    target_names=['Ham', 'Spam']))
        print("="*50 + "\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'confusion_matrix': cm
        }
        
    def predict(self, text):
        """
        Predict whether a message is spam
        
        Args:
            text: SMS message text
            
        Returns:
            Dictionary with prediction and probability
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess
        cleaned = clean_text(text)
        
        # TF-IDF
        tfidf_vec = self.tfidf.transform([cleaned])
        
        # SVD
        svd_vec = self.svd.transform(tfidf_vec)
        
        # Handcrafted features
        hand_vec = handcrafted_features([cleaned])
        
        # Combine and scale
        X_final = np.hstack([svd_vec, hand_vec])
        X_final = self.scaler.transform(X_final)
        
        # Predict
        prob_spam = self.classifier.predict_proba(X_final)[0][1]
        pred = 1 if prob_spam >= self.threshold else 0
        
        return {
            'text': text,
            'label': 'SPAM' if pred == 1 else 'HAM',
            'probability': prob_spam,
            'is_spam': bool(pred)
        }
    
    def save_model(self, filepath='models/spam_detector.pkl'):
        """
        Save the trained model and components
        
        Args:
            filepath: Path to save the model
        """
        if self.classifier is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'tfidf': self.tfidf,
            'svd': self.svd,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/spam_detector.pkl'):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf = model_data['tfidf']
        self.svd = model_data['svd']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.threshold = model_data['threshold']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    detector = SMSSpamDetector()
    detector.load_data('../data/spam.csv')
    detector.preprocess_and_extract_features()
    detector.train()
    detector.evaluate()
    
    # Test prediction
    test_message = "Congratulations! You've won a free prize. Click here now!"
    result = detector.predict(test_message)
    print(f"\nTest Message: {result['text']}")
    print(f"Prediction: {result['label']} (probability: {result['probability']:.4f})")