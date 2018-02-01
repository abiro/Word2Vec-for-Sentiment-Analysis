#!/usr/bin/env python

import json
import logging
import os
import sys

from sentiment_analysis.models.decision_tree_classifier import DecisionTreeClassifierModel  # noqa 501
from sentiment_analysis.models.logistic_regression import LogisticRegressionModel  # noqa 501
from sentiment_analysis.models.random_forest_classifier import RandomForestClassifierModel  # noqa 501
from sentiment_analysis.models.svc import SVCModel


def _train_and_score_model(
        cls,
        training_dataset_path,
        training_feature_vectors_path,
        validation_dataset_path,
        validation_feature_vectors_path,
        testing_dataset_path,
        testing_feature_vectors_path,
        predictions_dir):
    model = cls()
    model.train(training_dataset_path, training_feature_vectors_path)
    score = model.evaluate(validation_dataset_path,
                           validation_feature_vectors_path)
    logging.info('{} ROC AUC score: {}'.format(cls.__name__, score))
    prediction_out_path = os.path.join(predictions_dir, cls.__name__ + '.tsv')
    model.save_predict(testing_dataset_path, testing_feature_vectors_path,
                       prediction_out_path)
    params_str = json.dumps(model.get_params(), indent=2, sort_keys=True)
    logging.info('{} parameters:\n{}\n'.format(cls.__name__, params_str))


if len(sys.argv) != 8:
    logging.error('Expected 7 arguments: ' +
                  'training_dataset_path, validation_dataset_path, ' +
                  'testing_dataset_path, training_feature_vectors_path, ' +
                  'validation_feature_vectors_path, ' +
                  'testing_feature_vectors_path, predictions_dir')
    sys.exit(1)

training_dataset_path = sys.argv[1]
validation_dataset_path = sys.argv[2]
testing_dataset_path = sys.argv[3]
training_feature_vectors_path = sys.argv[4]
validation_feature_vectors_path = sys.argv[5]
testing_feature_vectors_path = sys.argv[6]
predictions_dir = sys.argv[7]
logging.info('training_dataset_path: {}'.format(training_dataset_path))
logging.info('validation_dataset_path: {}'.format(validation_dataset_path))
logging.info('testing_dataset_path: {}'.format(testing_dataset_path))
logging.info('training_feature_vectors_path: {}'.format(
    training_feature_vectors_path))
logging.info('validation_feature_vectors_path: {}'.format(
    validation_feature_vectors_path))
logging.info('testing_feature_vectors_path: {}'.format(
    testing_feature_vectors_path))
logging.info('predictions_dir: {}'.format(predictions_dir))

_train_and_score_model(DecisionTreeClassifierModel, training_dataset_path,
                       training_feature_vectors_path, validation_dataset_path,
                       validation_feature_vectors_path, testing_dataset_path,
                       testing_feature_vectors_path, predictions_dir)
_train_and_score_model(LogisticRegressionModel, training_dataset_path,
                       training_feature_vectors_path, validation_dataset_path,
                       validation_feature_vectors_path, testing_dataset_path,
                       testing_feature_vectors_path, predictions_dir)
_train_and_score_model(RandomForestClassifierModel, training_dataset_path,
                       training_feature_vectors_path, validation_dataset_path,
                       validation_feature_vectors_path, testing_dataset_path,
                       testing_feature_vectors_path, predictions_dir)
_train_and_score_model(SVCModel, training_dataset_path,
                       training_feature_vectors_path, validation_dataset_path,
                       validation_feature_vectors_path, testing_dataset_path,
                       testing_feature_vectors_path, predictions_dir)
