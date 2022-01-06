"""
Custom wrapper for Azure Text Analytics API.
https://docs.microsoft.com/en-us/azure/cognitive-services/language-service/sentiment-opinion-mining/overview
"""
from typing import Any
import numpy as np
import json

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from sklearn.base import BaseEstimator, ClassifierMixin


class CustomAzureTextAnalyticsClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn classifier implemented as a wrapper of
    Azure Cognitive Service for Language's API : Sentiment analysis.
    """

    classes_ = ("NEGATIVE", "POSITIVE")
    batch_size = 10
    cache = {}

    def __init__(self, endpoint: str, key: str, **kwargs) -> None:
        """
        Initialize the Azure Text Analytics client.

        Args:
            endpoint (str): Azure Cognitive Service endpoint.
            key (str): Azure Cognitive Service key.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.kwargs = kwargs
        client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        self.classifier = client.analyze_sentiment

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> ClassifierMixin:
        """
        Fill the predictions cache.

        Args:
            X (np.ndarray): Array of text documents.
            y (np.ndarray, optional): Array of true labels. Defaults to None.

        Returns:
            self: The current instance.
        """

        for i in range(0, len(X), self.batch_size):
            # We can only submit a batch of 10 documents at a time.
            batch = X[i : i + self.batch_size]
            results = self.classifier(batch)

            for idx, res in enumerate(results):
                self.cache[batch[idx]] = {
                    "positive": res.confidence_scores.positive,
                    "neutral": res.confidence_scores.neutral,
                    "negative": res.confidence_scores.negative,
                }

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
            or a request is sent to Azure and the result is saved in cache.
        Label as POSITIVE if the API's "positive" score is higher than the
            "negative" score.

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
                res = self.classifier([text])[0]
                self.cache[text] = {
                    "positive": res.confidence_scores.positive,
                    "neutral": res.confidence_scores.neutral,
                    "negative": res.confidence_scores.negative,
                }

            negative_proba = self._get_negative_proba_from_result(
                self.cache[text]
            )
            positive_proba = self._get_positive_proba_from_result(
                self.cache[text]
            )

            preds.append(
                "POSITIVE" if positive_proba > negative_proba else "NEGATIVE"
            )

        # Return the array of predicted labels
        return np.array(preds)

    def predict_proba(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Predict the class probabilities of the input samples.

        Sentiment analysis results are loaded from cache if present
            or a request is sent to Azure and the result is saved in cache.

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
                res = self.classifier([text])[0]
                self.cache[text] = {
                    "positive": res.confidence_scores.positive,
                    "neutral": res.confidence_scores.neutral,
                    "negative": res.confidence_scores.negative,
                }

            negative_proba = self._get_negative_proba_from_result(
                self.cache[text]
            )
            positive_proba = 1 - negative_proba

            preds.append((negative_proba, positive_proba))

        # Return the array of probabilities
        return np.array(preds)

    def _get_negative_proba_from_result(self, result: Any) -> float:
        """Get the negative class proba from the result.

        Half of the "neutral" score is added to the "negative" score.

        Args:
            result (Any): result from classifier.

        Raises:
            ValueError: There must be a negative class in the result.

        Returns:
            float: Negative class probability.
        """
        return result["negative"] + result["neutral"] / 2

    def _get_positive_proba_from_result(self, result: Any) -> float:
        """Get the positive class proba from the result.

        Half of the "neutral" score is added to the "positive" score.

        Args:
            result (Any): result from classifier.

        Raises:
            ValueError: There must be a positive class in the result.

        Returns:
            float: Positive class probability.
        """
        return result["positive"] + result["neutral"] / 2
