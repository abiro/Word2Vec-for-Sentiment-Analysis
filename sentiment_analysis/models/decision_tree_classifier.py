#!/usr/bin/env python

from sklearn.tree import DecisionTreeClassifier

from sentiment_analysis.models.model_base import ModelBase


class DecisionTreeClassifierModel(ModelBase):
    def __init__(self):
        self._clf = DecisionTreeClassifier()

    @property
    def _model(self):
        return self._clf
