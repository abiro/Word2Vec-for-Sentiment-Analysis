#!/usr/bin/env python

import logging

from sklearn.svm import SVC

from sentiment_analysis.models.model_base import ModelBase


class SVCModel(ModelBase):
    def __init__(self):
        self._clf = SVC(probability=True)

    @property
    def _model(self):
        return self._clf
