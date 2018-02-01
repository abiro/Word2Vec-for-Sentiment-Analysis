#!/usr/bin/env python

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import csv
import logging

import pandas as pd
from sklearn.metrics import roc_auc_score

from sentiment_analysis.make_dataset import read_dataset
from sentiment_analysis.make_dataset import write_dataset
from sentiment_analysis.utils import load_pickle
from sentiment_analysis.utils import write_pickle


class ModelBase:
    """Abstract base class for classification models"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def _model(self):
        """Returns a scikit-learn classification model."""
        pass

    def _get_feature_vectors(self, feature_vectors_path):
        return load_pickle(feature_vectors_path)

    def _get_labels(self, dataset_path):
        dataset = read_dataset(dataset_path)
        return dataset['sentiment'].values

    def get_params(self):
        return self._model.get_params()

    def evaluate(self, validation_dataset_path,
                 validation_feature_vectors_path):
        """Test the model's performance on the validation dataset."""
        logging.info('{} evaluating...'.format(self.__class__.__name__))
        feature_vectors = self._get_feature_vectors(
            validation_feature_vectors_path)
        labels = self._get_labels(validation_dataset_path)
        probabilities = self._model.predict_proba(feature_vectors)
        # Slice gets probabilities for label 1
        return roc_auc_score(labels, probabilities[:, 1])

    def predict(self, feature_vectors_path):
        """Produce predictions on the test set."""
        logging.info('{} predicting...'.format(self.__class__.__name__))
        feature_vectors = self._get_feature_vectors(feature_vectors_path)
        return self._model.predict(feature_vectors)

    def save_pickle(self, out_path):
        write_pickle(self, out_path)

    def save_predict(self, testing_dataset_path, feature_vectors_path,
                     prediction_df_out_path):
        """Save predictions for the test set."""
        logging.info('{} saving predictions...'.format(
            self.__class__.__name__))

        dataset = read_dataset(testing_dataset_path)
        prediction = self.predict(feature_vectors_path)

        out_data = {
            'id': dataset['id'],
            'sentiment': prediction
        }
        out_df = pd.DataFrame(data=out_data)
        logging.info('Writing predictions to: {}'.format(
            prediction_df_out_path))
        out_df.to_csv(prediction_df_out_path, quoting=csv.QUOTE_NONE,
                      index=False)

    def train(self, training_dataset_path, training_feature_vectors_path):
        logging.info('{} training...'.format(self.__class__.__name__))
        feature_vectors = self._get_feature_vectors(
            training_feature_vectors_path)
        labels = self._get_labels(training_dataset_path)
        self._model.fit(feature_vectors, labels)
