#!/usr/bin/env python

import logging
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sentiment_analysis.make_dataset import read_dataset
from sentiment_analysis.utils import write_pickle


def make_random_embedding_feature_vectors(
        corpus_dataset_path,
        training_dataset_path,
        validation_dataset_path,
        testing_dataset_path,
        training_term_matrix_out_path,
        validation_term_matrix_out_path,
        testing_term_matrix_out_path,
        embedding_size=200):
    """Create feature vectors from randomized word embedding vectors.

    The feature vectors are produced by taking the mean of the word embeddings
    in a review.

    Arguments:
        corpus_dataset_path: The path to the TSV file holding the dataset from
            which the vocabulary should be learned.
        training_dataset_path: The path to the TSV file holding the training
            dataset.
        validation_dataset_path: The path to the TSV file holding the
            validation dataset.
        testing_dataset_path: The path to the TSV file holding the testing
            dataset.
        training_term_matrix_out_path: The path to save the training
            term-document matrix to as a pickle file.
        validation_term_matrix_out_path: The path to save the validation term-
            document matrix to as a pickle file.
        testing_term_matrix_out_path: The path to save the testing term-
            document matrix to as a pickle file.
        embedding_size: The dimensionality of the embedding vectors.
    """
    corpus_dataset = read_dataset(corpus_dataset_path)

    vectorizer = CountVectorizer(analyzer='word')
    # Learn the vocabualry.
    vectorizer.fit(corpus_dataset['review'].values)
    embeddings = {}
    for word in vectorizer.get_feature_names():
        # Create a random but normalized vector as word embedding.
        embeddings[word] = _get_norm_random_vector(embedding_size)

    training_dataset = read_dataset(training_dataset_path)
    validation_dataset = read_dataset(validation_dataset_path)
    testing_dataset = read_dataset(testing_dataset_path)

    training_term_matrix = _make_feature_vectors(embeddings,
        training_dataset['review'].values, embedding_size)
    validation_term_matrix = _make_feature_vectors(embeddings,
        validation_dataset['review'].values, embedding_size)
    testing_term_matrix = _make_feature_vectors(embeddings,
        testing_dataset['review'].values, embedding_size)

    write_pickle(training_term_matrix, training_term_matrix_out_path)
    write_pickle(validation_term_matrix, validation_term_matrix_out_path)
    write_pickle(testing_term_matrix, testing_term_matrix_out_path)


def _make_feature_vectors(embeddings, reviews, dim):
    feature_vectors = []
    for r in reviews:
        vectors = []
        for word in r.split(' '):
            try:
                vectors.append(embeddings[word])
            except KeyError:
                # Ignore unknown words
                pass
        if len(vectors) > 0:
            feature_vectors.append(_get_mean_vector(vectors))
        else:
            feature_vectors.append(_get_norm_random_vector(dim))
    return np.array(feature_vectors)


def _get_mean_vector(vectors):
    if len(vectors) == 0:
        raise ValueError('Expected at least 1 vector')
    return np.sum(vectors, axis=0) / len(vectors)


def _get_norm_random_vector(dim):
    rand_vector = np.random.rand(dim)
    # Normalize random vector
    return rand_vector / np.linalg.norm(rand_vector)


if __name__ == '__main__':
    """Create term matrices for the training, testing and validation sets using
    the unlabeled datatest to learn the vocabulary.

    Arguments:
        unlabeled_dataset_path: The path to the TSV file holding the unlabeled
            dataset.
        training_dataset_path: The path to the TSV file holding the training
            dataset.
        validation_dataset_path: The path to the TSV file holding the
            validation dataset.
        testing_dataset_path: The path to the TSV file holding the testing
            dataset.
        training_term_matrix_out_path: The path to save the training
            term-document matrix to as a pickle file.
        validation_term_matrix_out_path: The path to save the validation term-
            document matrix to as a pickle file.
        testing_term_matrix_out_path: The path to save the testing term-
            document matrix to as a pickle file.
    """
    if len(sys.argv) != 8:
        logging.error('Expected 7 arguments: unlabeled_dataset_path, ' +
                      'training_dataset_path, validation_dataset_path, ' +
                      'testing_dataset_path, training_term_matrix_out_path, ' +
                      'validation_term_matrix_out_path, ' +
                      'testing_term_matrix_out_path')
        sys.exit(1)

    unlabeled_dataset_path = sys.argv[1]
    training_dataset_path = sys.argv[2]
    validation_dataset_path = sys.argv[3]
    testing_dataset_path = sys.argv[4]
    training_term_matrix_out_path = sys.argv[5]
    validation_term_matrix_out_path = sys.argv[6]
    testing_term_matrix_out_path = sys.argv[7]

    logging.info('unlabeled_dataset_path: {}'.format(unlabeled_dataset_path))
    logging.info('training_dataset_path: {}'.format(training_dataset_path))
    logging.info('validation_dataset_path: {}'.format(validation_dataset_path))
    logging.info('testing_dataset_path: {}'.format(testing_dataset_path))
    logging.info('training_term_matrix_out_path: {}'.format(
        training_term_matrix_out_path))
    logging.info('validation_term_matrix_out_path: {}'.format(
        validation_term_matrix_out_path))
    logging.info('testing_term_matrix_out_path: {}'.format(
        testing_term_matrix_out_path))

    make_random_embedding_feature_vectors(
        unlabeled_dataset_path,
        training_dataset_path,
        validation_dataset_path,
        testing_dataset_path,
        training_term_matrix_out_path,
        validation_term_matrix_out_path,
        testing_term_matrix_out_path)
