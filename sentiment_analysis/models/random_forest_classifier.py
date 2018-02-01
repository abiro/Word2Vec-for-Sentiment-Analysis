#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier

from sentiment_analysis.models.model_base import ModelBase


class RandomForestClassifierModel(ModelBase):
    def __init__(self):
        self._clf = RandomForestClassifier()

    @property
    def _model(self):
        return self._clf
