#!/usr/bin/env python

from __future__ import print_function

import csv
import logging
import os
import re
import sys

from bs4 import BeautifulSoup
import pandas as pd


RANDOM_SEED = 64


def clean_review(review):
    """Turn a raw HTML review into a string of cleaned up words.

    Source: Kaggle tutorial https://github.com/wendykan/DeepLearningMovies

    Arguments:
        review: An HTML string.
    Returns:
        A string that contains the cleaned up words separated by spaces.
    """
    # Remove HTML
    review_text = BeautifulSoup(review, 'html.parser').get_text()

    # Remove non-letters
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)

    # Convert words to lower case and split them
    words = review_text.lower().split()

    # Return a string of words separated by spaces.
    return ' '.join(words)


def clean_dataset(dataset_path, out_path):
    """Transform a review dataset by transforming each review to a cleaned up
        list of words.

    The results are written to the desired location as a TSV file with the
    same structure as the input.

    Based on this: Kaggle tutorial:
        https://github.com/wendykan/DeepLearningMovies

    Arguments:
        dataset_path: The absolute path of the TSV dataset that must have a
            "review" column.
        out_path: The absolute path of the TSV outfile.
    """
    dataset = read_dataset(dataset_path)

    cleaned_reviews = dataset['review'].apply(clean_review)
    clean_dataset = dataset.assign(review=cleaned_reviews)
    write_dataset(clean_dataset, out_path)


def make_text_file(dataset_path, out_path):
    """Turn a tsv file with reviews into a text file with one review per line.

    Arguments:
        dataset_path: The absolute path of the TSV dataset that must have a
            "review" column.
        out_path: The absolute path of the resulting text file.
    """
    dataset = read_dataset(dataset_path)
    logging.info('Writing reviews to textfile: {}'.format(out_path))
    with open(out_path, 'w') as outfile:
        for line in dataset['review'].values:
            print(line, file=outfile)


def read_dataset(dataset_path):
    """Load a TSV dataset from disk.

    Arguments:
        dataset_path: The absolute path to the dataset.
    Returns:
        A pandas.DataFrame instance.
    """
    logging.info('Reading dataset from: {}'.format(dataset_path))
    return pd.read_csv(dataset_path, header=0, delimiter='\t',
                       quoting=csv.QUOTE_NONE)


def training_validation_split(
        dataset_path,
        training_out_path,
        validation_out_path,
        validation_ratio=0.2,
        random_seed=None):
    """Split a dataset into training and validation sets.

    Arguments:
        dataset_path: Absolute path to the TSV file with the data.
        training_out_path: Absolute path to the TSV file that will hold
            the training part of the split.
        validation_out_path: Absolute path to the TSV file that will hold
            the validation part of the split.
        validation_ratio: The ratio of the original dataset that is to be used
            as validation data. Optional, defaults to 0.2.
        random_seed: The integer seed of the random generator that is to be
            used for the split. Defaults to None in which case no seed is set.
    """
    dataset = read_dataset(dataset_path)
    logging.info(('Training validation split params:\n  validation_ratio: {}' +
                  '\n  random_seed: {}').format(validation_ratio, random_seed))
    if random_seed is not None:
        validation_set = dataset.sample(frac=validation_ratio,
                                        random_state=random_seed)
    else:
        validation_set = dataset.sample(frac=validation_ratio)

    training_set = dataset.drop(validation_set.index)
    write_dataset(training_set, training_out_path)

    write_dataset(validation_set, validation_out_path)


def write_dataset(dataset, out_path):
    """Write a TSV dataset to disk.

    Arguments:
        dataset: A pandas.DataFrame instance.
        out_path: The absolute path of the result.
    """
    logging.info('Writing dataset to: {}'.format(out_path))
    dataset.to_csv(out_path, sep='\t', quoting=csv.QUOTE_NONE, index=False)


if __name__ == '__main__':
    """Clean up the source dataset from Kaggle and create a validation set.
        https://www.kaggle.com/c/word2vec-nlp-tutorial/data

    Arguments:
        input_dir: The directory holding the source dataset.
        labeled_filename: The filename of the labeled training dataset.
        unlabeled_filename: The filename of the unlabeled training dataset.
        testing_filename: The filename of the testing dataset.
        out_dir: The directory to write the cleaned results to.
        labeled_training_filename: The filename of the labeled training set
            after the training-validation split.
        labeled_validation_filename: The filename of the labeled validation set
            after the training-validation split.
    """
    if len(sys.argv) != 8:
        logging.error('Expected 7 arguments: input_dir labeled_filename ' +
                      'unlabeled_filename testing_filename out_dir ' +
                      'labeled_training_filename labeled_validation_filename')
        sys.exit(1)

    input_dir = sys.argv[1]
    dataset_filenames = sys.argv[2:5]
    out_dir = sys.argv[5]
    labeled_training_filename = sys.argv[6]
    labeled_validation_filename = sys.argv[7]

    logging.info('input_dir: {}'.format(input_dir))
    logging.info(('labeled_filename, unlabeled_filename, ' +
                  'testing_filename: {}'.format(dataset_filenames)))
    logging.info('out_dir: {}'.format(out_dir))
    logging.info('labeled_training_filename: {}'.format(
        labeled_training_filename))
    logging.info('labeled_validation_filename: {}'.format(
        labeled_validation_filename))

    for filename in dataset_filenames:
        dataset_path = os.path.join(input_dir, filename)
        tsv_out_path = os.path.join(out_dir, filename)
        clean_dataset(dataset_path, tsv_out_path)
        text_out_path = os.path.join(out_dir, filename.replace('.tsv', '.txt'))
        make_text_file(tsv_out_path, text_out_path)

    clean_labeled_path = os.path.join(out_dir, dataset_filenames[0])
    labeled_training_path = os.path.join(out_dir, labeled_training_filename)
    labeled_validation_path = os.path.join(out_dir,
                                           labeled_validation_filename)
    training_validation_split(clean_labeled_path, labeled_training_path,
                              labeled_validation_path, random_seed=RANDOM_SEED)
