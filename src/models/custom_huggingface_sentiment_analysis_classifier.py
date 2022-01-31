"""
Custom wrapper for HuggingFace Sequence Classification.
https://huggingface.co/docs/transformers/master/en/task_summary#sequence-classification
"""
import json
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import pipeline


class CustomHuggingfaceSentimentAnalysisClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn classifier implemented as a wrapper of
    HuggingFace's Sequence Classification task : Sentiment analysis.
    """

    classes_ = ("NEGATIVE", "POSITIVE")
    cache: Dict[str, Any] = {}

    def __init__(self, **kwargs) -> None:
        """
        Initialize the HuggingFace classifier.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.kwargs = kwargs
        self.classifier = pipeline("sentiment-analysis")

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> ClassifierMixin:
        """
        Fill the predictions cache.

        Args:
            X (np.ndarray): Array of text documents.
            y (np.ndarray, optional): Array of true labels. Defaults to None.

        Returns:
            self: The current instance.
        """
        self.cache = {text: self.classifier(text) for text in X}

        return self

    def save_cache_json(self, filename: str) -> None:
        """
        Save the cache in a JSON file.

        Args:
            filename (str): Name of the JSON file.

        Returns:
            None
        """
        with open(filename, "w") as f:
            json.dump(self.cache, f)

    def load_cache_json(self, filename: str) -> None:
        """
        Load the cache from a JSON file.

        Args:
            filename (str): Name of the JSON file.

        Returns:
            None
        """
        with open(filename, "r") as f:
            self.cache = json.load(f)

    def predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Predict the class of the input samples.

        Sentiment analysis results are loaded from cache if present
            or a compute with HuggingFace and the result is saved in cache.

        Args:
            X (np.ndarray): Array of text documents.
            y (np.ndarray, optional): Array of true labels. Defaults to None.

        Returns:
            np.ndarray: Predicted classes.
        """
        # Store predictions in a list
        preds = []

        for text in X:
            if text not in self.cache.keys():
                self.cache[text] = self.classifier(text)

            preds.append(self.cache[text][0]["label"])

        # Return the array of predicted labels
        return np.array(preds)

    def predict_proba(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Predict the class probabilities of the input samples.

        Sentiment analysis results are loaded from cache if present
            or compute with HuggingFace and the result is saved in cache.

        Args:
            X (np.ndarray): Array of text documents.
            y (np.ndarray, optional): Array of true labels. Defaults to None.

        Returns:
            np.ndarray: Array of predicted classes probabilities (NEGATIVE, POSITIVE).
        """
        # Store predictions in a list
        preds = []

        for text in X:
            if text not in self.cache.keys():
                self.cache[text] = self.classifier(text)

            negative_proba = self._get_negative_proba_from_result(self.cache[text])
            positive_proba = 1 - negative_proba

            preds.append((negative_proba, positive_proba))

        # Return the array of probabilities
        return np.array(preds)

    def _get_negative_proba_from_result(self, result: list[dict[str, Any]]) -> float:
        """Get the negative class proba from the result.

        Args:
            result (list[dict[str, Any]]): result from classifier.

        Returns:
            float: Negative class probability.
        """
        if result[0]["label"] == "NEGATIVE":
            return result[0]["score"]
        else:
            return 1 - result[0]["score"]

    def _get_positive_proba_from_result(self, result: list[dict[str, Any]]) -> float:
        """Get the positive class proba from the result.

        Args:
            result (list[dict[str, Any]]): result from classifier.

        Returns:
            float: Positive class probability.
        """
        if result[0]["label"] == "POSITIVE":
            return result[0]["score"]
        else:
            return 1 - result[0]["score"]
