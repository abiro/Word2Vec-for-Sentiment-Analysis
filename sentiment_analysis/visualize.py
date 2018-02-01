#!/usr/bin/env python

import logging
from operator import itemgetter
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sentiment_analysis.make_dataset import read_dataset


def make_word_count_historigram(dataset_path, out_dir, n_bins):
    """Create a historigram of words counts in reviews.

    dataset_path: The path to the TSV file holding the cleaned dataset from
        that should be analyzed.
    out_dir: The path to the directory where the plot should be saved.
    """
    df = read_dataset(dataset_path)
    mapper = np.vectorize(lambda r: len(r.split(' ')))
    review_wordcount = mapper(df['review'].values)
    logging.info('Max word count: {}'.format(review_wordcount.max()))
    plt.cla()
    plt.title('Histogram of word counts in reviews')
    plt.xlabel('Word count')
    plt.ylabel('N reviews')
    plt.gca().set_yscale('log', nonposy='clip')
    plt.grid(True)
    count, bins, ignored = plt.hist(review_wordcount, n_bins,
                                    histtype='stepfilled')
    out_path = os.path.join(out_dir, 'word_count_historigram.png')
    logging.info('Saving figure to {}'.format(out_path))
    plt.savefig(out_path, dpi=300)


def make_word_frequency_historigram(dataset_path, out_dir, n_bins):
    """Create a historigram of words frequencies in the reviews.

    dataset_path: The path to the TSV file holding the cleaned dataset from
        that should be analyzed.
    out_dir: The path to the directory where the plot should be saved.
    """
    df = read_dataset(dataset_path)
    vectorizer = CountVectorizer(analyzer='word', stop_words='english')
    vectorizer.fit(df['review'].values)
    X = vectorizer.transform(df['review'].values)
    # From https://stackoverflow.com/a/16078639/2650622
    frequencies = np.asarray(X.sum(axis=0)).ravel()
    word_freq_pairs = zip(vectorizer.get_feature_names(), frequencies)
    word_freq_pairs.sort(key=itemgetter(1), reverse=True)
    logging.info('Vocabulary size: {}'.format(len(word_freq_pairs)))
    logging.info('Top 10 most frequent words excluding English stop words')
    for word, freq in word_freq_pairs[:10]:
        logging.info('{}: {}'.format(word, freq))
    plt.cla()
    plt.title('Histogram of word frequencies in reviews without English '
              'stop words')
    plt.xlabel('Word frequencies')
    plt.ylabel('N words')
    plt.gca().set_yscale('log', nonposy='clip')
    plt.gca().set_xscale('log', nonposy='clip')
    plt.grid(True)
    count, bins, ignored = plt.hist(frequencies, n_bins,
                                    histtype='stepfilled')
    out_path = os.path.join(out_dir, 'word_freq_historigram.png')
    logging.info('Saving figure to {}'.format(out_path))
    plt.savefig(out_path, dpi=300)


if __name__ == '__main__':
    """Create plots for the dataset.

    dataset_path: The path to the TSV file holding the cleaned dataset from
        that should be analyzed.
    out_dir: The path to the directory where the plot should be saved.
    """
    if len(sys.argv) != 3:
        logging.error('Expected 2 arguments: dataset_path, out_dir')
        sys.exit(1)

    dataset_path = sys.argv[1]
    training_dataset_path = sys.argv[2]

    logging.info('dataset_path: {}'.format(dataset_path))
    logging.info('training_dataset_path: {}'.format(training_dataset_path))

    make_word_count_historigram(dataset_path, training_dataset_path, 100)
    make_word_frequency_historigram(dataset_path, training_dataset_path, 10000)
