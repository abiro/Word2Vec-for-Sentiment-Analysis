#!/usr/bin/env python

import logging
import sys

from sklearn.feature_extraction.text import CountVectorizer

from sentiment_analysis.make_dataset import read_dataset
from sentiment_analysis.utils import write_pickle


def make_bag_of_words_features(
        corpus_dataset_path,
        training_dataset_path,
        validation_dataset_path,
        testing_dataset_path,
        training_term_matrix_out_path,
        validation_term_matrix_out_path,
        testing_term_matrix_out_path,
        max_features=5000):
    """Create a term-document matrix using the bag of words method.

    Stop words are removed from the vocabulary.

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
        max_features: The maximum dimensionality of the feature vectors.
    """
    corpus_dataset = read_dataset(corpus_dataset_path)

    # Remove english stop words from the vocabulary.
    vectorizer = CountVectorizer(analyzer='word', max_features=max_features,
                                 stop_words='english')

    # Learn the vocabualry.
    vectorizer.fit(corpus_dataset['review'].values)

    training_dataset = read_dataset(training_dataset_path)
    validation_dataset = read_dataset(validation_dataset_path)
    testing_dataset = read_dataset(testing_dataset_path)

    training_term_matrix = vectorizer.transform(
        training_dataset['review'].values)
    validation_term_matrix = vectorizer.transform(
        validation_dataset['review'].values)
    testing_term_matrix = vectorizer.transform(
        testing_dataset['review'].values)

    write_pickle(training_term_matrix, training_term_matrix_out_path)
    write_pickle(validation_term_matrix, validation_term_matrix_out_path)
    write_pickle(testing_term_matrix, testing_term_matrix_out_path)


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

    make_bag_of_words_features(
        unlabeled_dataset_path,
        training_dataset_path,
        validation_dataset_path,
        testing_dataset_path,
        training_term_matrix_out_path,
        validation_term_matrix_out_path,
        testing_term_matrix_out_path)
