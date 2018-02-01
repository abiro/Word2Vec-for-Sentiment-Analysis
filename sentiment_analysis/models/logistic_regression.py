#!/usr/bin/env python

from sklearn.linear_model import LogisticRegressionCV

from sentiment_analysis.models.model_base import ModelBase


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        self._clf = LogisticRegressionCV()

    @property
    def _model(self):
        return self._clf
