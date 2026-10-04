"""
Machine Learning Pipeline for AI Email Spam Detector.
Implements TF-IDF vectorization, Multinomial Naive Bayes, Logistic Regression,
and model evaluation metrics.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.preprocessing import preprocess_text


class SpamClassifier:
    """
    NLP & ML Classifier for Spam Detection using TF-IDF and Naive Bayes / Logistic Regression.
    """

    def __init__(self, algorithm: str = "naive_bayes"):
        self.algorithm_name = algorithm
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model = None
        self.metrics: Dict[str, Any] = {}
        self.is_trained: bool = False
        self.data_df: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []

    def _init_classifier(self):
        if self.algorithm_name == "logistic_regression":
            return LogisticRegression(C=1.0, max_iter=200, random_state=42)
        # Default is Multinomial Naive Bayes
        return MultinomialNB(alpha=0.5)

    def load_data(self, data_path: str = "data/spam_dataset.csv") -> pd.DataFrame:
        """
        Loads and validates the spam/ham dataset.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at '{data_path}'. Please check the path.")
        
        df = pd.read_csv(data_path)
        if "label" not in df.columns or "text" not in df.columns:
            raise ValueError("Dataset must contain 'label' and 'text' columns.")

        # Clean invalid or missing rows
        df = df.dropna(subset=["label", "text"])
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["text"] = df["text"].astype(str)
        self.data_df = df
        return df

    def train(
        self,
        data_path: str = "data/spam_dataset.csv",
        test_size: float = 0.25,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Trains the TF-IDF vectorizer and classifier, evaluating on a test split.
        """
        df = self.load_data(data_path)

        # Preprocess dataset texts
        df["processed_text"] = df["text"].apply(preprocess_text)
        
        # Binary target: 1 for spam, 0 for ham
        df["target"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

        X = df["processed_text"]
        y = df["target"]

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Initialize TF-IDF Vectorizer with unigrams & bigrams
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        # Initialize and fit model
        self.model = self._init_classifier()
        self.model.fit(X_train_vec, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_samples": int(len(df)),
            "algorithm": "Multinomial Naive Bayes" if self.algorithm_name == "naive_bayes" else "Logistic Regression"
        }

        self.is_trained = True
        return self.metrics

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Classifies a single message as SPAM or NOT SPAM.
        Returns label, confidence percentage, probabilities, and extracted feature terms.
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise RuntimeError("Model has not been trained yet. Call train() before predict().")

        if not isinstance(raw_text, str) or not raw_text.strip():
            return {
                "prediction": "UNKNOWN",
                "label": "empty",
                "confidence": 0.0,
                "spam_prob": 0.0,
                "ham_prob": 0.0,
                "top_terms": [],
                "processed_text": ""
            }

        processed = preprocess_text(raw_text)
        
        # If text is empty after preprocessing (e.g. only symbols or stopwords)
        if not processed.strip():
            # Fallback evaluation based on presence of raw markers (e.g., links or symbols)
            processed = raw_text.lower().strip()

        vec_input = self.vectorizer.transform([processed])
        
        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec_input)[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            pred_val = self.model.predict(vec_input)[0]
            spam_prob = 1.0 if pred_val == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        # Determine final label and confidence
        if spam_prob >= 0.50:
            prediction = "SPAM"
            confidence = spam_prob * 100
        else:
            prediction = "NOT SPAM"
            confidence = ham_prob * 100

        # Extract top TF-IDF words in this document
        top_terms = self._get_contributing_terms(vec_input, max_terms=8)

        return {
            "prediction": prediction,
            "label": "spam" if prediction == "SPAM" else "ham",
            "confidence": round(confidence, 2),
            "spam_prob": round(spam_prob * 100, 2),
            "ham_prob": round(ham_prob * 100, 2),
            "top_terms": top_terms,
            "processed_text": processed
        }

    def _get_contributing_terms(self, vec_input, max_terms: int = 8) -> List[str]:
        """
        Identifies highest weighted TF-IDF tokens present in the current input.
        """
        if self.vectorizer is None:
            return []

        row_indices = vec_input.nonzero()[1]
        feature_array = np.array(self.feature_names)
        
        if len(row_indices) == 0:
            return []

        tfidf_scores = vec_input.data
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        
        top_features = []
        for idx in sorted_indices[:max_terms]:
            feat_idx = row_indices[idx]
            top_features.append(feature_array[feat_idx])

        return top_features
