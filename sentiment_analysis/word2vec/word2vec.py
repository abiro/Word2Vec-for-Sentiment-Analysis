# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Modified version of:
https://github.com/tensorflow/models/blob/56765e5b60f998a8e7023c3f088e657c0d25d1a4/tutorials/embedding/word2vec_optimized.py  # noqa 501

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient 
  using true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import tensorflow as tf

from sentiment_analysis.make_dataset import read_dataset
from sentiment_analysis.utils import write_pickle


file_path = os.path.dirname(os.path.realpath(__file__))
# TODO rename to word2vec ops
word2vec = tf.load_op_library(os.path.join(file_path, 'word2vec_ops.so'))

tf.app.flags.DEFINE_string('save_path', None, 'Directory to write the model.')
tf.app.flags.DEFINE_string(
    'train_data', None,
    'Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.')
tf.app.flags.DEFINE_string(
    'trained_model_meta', None, 'The trained model\'s meta file, eg. '
    'model.ckpt-1591664.meta')
tf.app.flags.DEFINE_string(
    'reviews_training_path', None, 'The path to the training dataset from '
    'which to create feature vectors')
tf.app.flags.DEFINE_string(
    'reviews_validation_path', None, 'The path to the validation dataset from '
    'which to create feature vectors')
tf.app.flags.DEFINE_string(
    'reviews_testing_path', None, 'The path to the testing dataset from '
    'which to create feature vectors')
tf.app.flags.DEFINE_string(
    'features_out_dir', None, 'The directory to save the created feature '
    'vectors to')
tf.app.flags.DEFINE_integer(
    'embedding_size',
    200,
    'The embedding dimension size.')
tf.app.flags.DEFINE_integer(
    'epochs_to_train', 15,
    'Number of epochs to train. Each epoch processes the training data once '
    'completely.')
tf.app.flags.DEFINE_float('learning_rate', 0.025, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('num_neg_samples', 25,
                            'Negative samples per training example.')
tf.app.flags.DEFINE_integer('batch_size', 500,
                            'Numbers of training examples each step processes '
                            '(no minibatching).')
tf.app.flags.DEFINE_integer('concurrent_steps', 12,
                            'The number of concurrent training steps.')
tf.app.flags.DEFINE_integer(
    'window_size',
    5,
    'The number of words to predict to the left and right '
    'of the target word.')
tf.app.flags.DEFINE_integer(
    'min_count',
    5,
    'The minimum number of word occurrences for it to be '
    'included in the vocabulary.')
tf.app.flags.DEFINE_float(
    'subsample',
    1e-3,
    'Subsample threshold for word occurrence. Words that appear '
    'with higher frequency will be randomly down-sampled. Set '
    'to 0 to disable.')
tf.app.flags.DEFINE_boolean(
    'interactive', False,
    'If true, enters an IPython interactive session to play with the trained '
    'model. ')

FLAGS = tf.app.flags.FLAGS


class Options(object):
    """Options used by our word2vec model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.

        # The training text file.
        self.train_data = FLAGS.train_data

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target
        # word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Eval options.

        # The text file for eval.
        # self.eval_data = FLAGS.eval_data

        self.trained_model_meta = FLAGS.trained_model_meta
        self.reviews_training_path = FLAGS.reviews_training_path
        self.reviews_validation_path = FLAGS.reviews_validation_path
        self.reviews_testing_path = FLAGS.reviews_testing_path
        self.features_out_dir = FLAGS.features_out_dir


class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, options, session, save_vocab=True):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        if save_vocab:
            self.save_vocab()
        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def build_graph(self):
        """Build the model graph."""
        opts = self._options

        # The training data. A text file.
        (words, counts, words_per_epoch, current_epoch, total_words_processed,
         examples, labels) = word2vec.skipgram_word2vec(
            filename=opts.train_data,
            batch_size=opts.batch_size,
            window_size=opts.window_size,
            min_count=opts.min_count,
            subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run(
            [words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        logging.info('Data file: {}'.format(opts.train_data))
        logging.info('Vocab size: {}, UNK'.format(opts.vocab_size - 1))
        logging.info('Words per epoch: '.format(opts.words_per_epoch))

        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        # Declare all variables we need.
        # Input words embedding: [vocab_size, emb_dim]
        w_in = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size,
                 opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
            name="w_in")

        # Global step: scalar, i.e., shape [].
        w_out = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

        # Global step: []
        global_step = tf.Variable(0, name="global_step")

        # Linear learning rate decay.
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001,
            1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

        # Training nodes.
        inc = global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            train = word2vec.neg_train_word2vec(
                w_in,
                w_out,
                examples,
                labels,
                lr,
                vocab_count=opts.vocab_counts.tolist(),
                num_negative_samples=opts.num_samples)

        self._w_in = w_in
        self._examples = examples
        self._labels = labels
        self._lr = lr
        self._train = train
        self.global_step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed
        self._counts = counts

    def build_eval_graph(self):
        """Build the evaluation graph."""
        # Eval graph
        opts = self._options

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self._w_in, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, opts.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        # Gather word vectors from the normalized embeddings.
        self._word_ids = tf.placeholder(dtype=tf.int32)
        self._word_vectors = tf.gather(nemb, self._word_ids)

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run(
            [self._epoch, self._words])

        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time = initial_words, time.time()
        while True:
            time.sleep(5)  # Reports our progress once a while.
            (epoch, step, words, lr) = self._session.run(
                [self._epoch, self.global_step, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (
                now - last_time)
            logging.info(
                'Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r' %
                (epoch, step, lr, rate))
            sys.stdout.flush()
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()

    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, 'vocab.txt'), 'w') as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(
                    opts.vocab_words[i]).encode('utf-8')
                f.write('%s %d\n' % (vocab_word,
                                     opts.vocab_counts[i]))

    def nearby(self, words, num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
            logging.info("\n%s\n=====================================" %
                         (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                logging.info(
                    "%-20s %6.4f" %
                    (self._id2word[neighbor], distance))

    def make_averaged_feature_vectors(self, dataset_path,
                                      feature_vectors_out_dir):
        """Make feature vectors for reviews in a dataset by averaging the
        word vectors in the review.

        The feature vectors are saved as a 2d numpy array where the first
        dimeansion corresponds to rows in the dataset.
        The "FeatureVectors.pickle" suffix is appended to the dataset file
        name to create the feature vectors file name.

        Arguments:
            dataset_path: The path to the TSV file holding the reviews dataset.
            feature_vectors_out_dir: The dir to save the term matrix to as a
                pickle file.
        """
        logging.info('Creating feature vectors from {}'.format(dataset_path))
        t0 = time.time()
        dataset = read_dataset(dataset_path)
        nemb = self.get_normalized_embeddings()
        n_rows = len(dataset.index)
        mean_vectors = []
        for index, row in dataset.iterrows():
            if index % 1000 == 0:
                logging.info('Processing row {}/{}'.format(index, n_rows))
            word_ids = []
            for w in row['review'].split(' '):
                try:
                    word_ids.append(self._word2id[w])
                except KeyError:
                    # Ignore unknown words
                    pass
            mean_vectors.append(self._get_mean_vector(nemb, word_ids))

        # Get file name from dataset_path
        dataset_file_name = os.path.splitext(os.path.basename(dataset_path))[0]
        feature_vec_file_name = dataset_file_name + 'FeatureVectors.pickle'
        feature_vec_out_path = os.path.join(feature_vectors_out_dir,
                                            feature_vec_file_name)
        write_pickle(np.array(mean_vectors), feature_vec_out_path)

        dt = time.time() - t0
        logging.info('Total time: {} seconds'.format(int(round(dt))))

    def make_clustered_feature_vectors(
            self,
            dataset_path,
            feature_vectors_out_dir,
            n_most_frequent=5000):
        """Make feature vectors for reviews in a dataset by clustering the word
        vectors and counting the number of words in the review in each cluster.

        The feature vectors are saved as a 2d numpy array where the first
        dimeansion corresponds to rows in the dataset.
        The "ClusteredFeatureVectors.pickle" suffix is appended to the dataset
        file name to create the feature vectors file name.
        The clustering is pickled to the directory as "clustering.pickle".

        Arguments:
            dataset_path: The path to the TSV file holding the reviews dataset.
            feature_vectors_out_dir: The dir to save the term matrix to as a
                pickle file.
            n_most_frequent: The number of most frequent words to include in
                the clustering from the vocabulary (excluding stop words and
                unknown). Defaults to 5000.
        """
        logging.info('Creating clustered feature vectors from {}'.format(
            dataset_path))
        t0 = time.time()
        opts = self._options
        dataset = read_dataset(dataset_path)
        nemb = self.get_normalized_embeddings()
        most_freq_ids = []
        most_freq_words = []
        i = 0
        while len(most_freq_ids) < n_most_frequent and i < opts.vocab_size:
            word = opts.vocab_words[i]
            if word != 'UNK' and word not in ENGLISH_STOP_WORDS:
                most_freq_ids.append(self._word2id[word])
                most_freq_words.append(word)
            i += 1

        affinities = -1.0 * cdist(nemb[most_freq_ids], nemb[most_freq_ids],
                                  'cosine')
        af = AffinityPropagation(affinity='precomputed')
        af.fit(affinities)
        n_clusters = len(af.cluster_centers_indices_)
        logging.info('Number of clusters: {}'.format(n_clusters))
        word2cluster = {}
        for i, word in enumerate(most_freq_words):
            word2cluster[word] = af.labels_[i]

        feature_vectors = []
        n_rows = len(dataset.index)
        for index, row in dataset.iterrows():
            if index % 1000 == 0:
                logging.info('Processing row {}/{}'.format(index, n_rows))
            fv = np.zeros(n_clusters)
            for w in row['review'].split(' '):
                try:
                    cluster_id = word2cluster[w]
                    fv[cluster_id] += 1
                except KeyError:
                    # Ignore unknown words
                    pass
            feature_vectors.append(fv)

        # Get file name from dataset_path
        dataset_file_name = os.path.splitext(os.path.basename(dataset_path))[0]
        feature_vec_file_name = (dataset_file_name +
                                 'ClusteredFeatureVectors.pickle')
        feature_vec_out_path = os.path.join(feature_vectors_out_dir,
                                            feature_vec_file_name)
        write_pickle(np.array(feature_vectors), feature_vec_out_path)
        clustering_out_path = os.path.join(
            feature_vectors_out_dir,
            dataset_file_name + 'Clustering.pickle')
        write_pickle(af, clustering_out_path)
        word2cluster = os.path.join(feature_vectors_out_dir,
                                    dataset_file_name + 'Word2cluster.pickle')
        write_pickle(af, word2cluster)

        dt = time.time() - t0
        logging.info('Total time: {} seconds'.format(int(round(dt))))

    def get_normalized_embeddings(self):
        return self._session.run(
            [self._word_vectors, self._word_ids],
            {self._word_ids: list(range(len(self._id2word)))})[0]

    def _get_mean_vector(self, nemb, word_ids):
        # TODO need to normalize?
        return self._get_sum_vector(nemb, word_ids) / len(word_ids)

    def _get_sum_vector(self, nemb, word_ids):
        # TODO use numpy built in instead?
        sum_vector = np.zeros(nemb.shape[1])
        for i in word_ids:
            sum_vector += nemb[i]
        return sum_vector


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def run_training():
    """Train a word2vec model."""
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            model = Word2Vec(opts, session)
        for _ in xrange(opts.epochs_to_train):
            # Process one epoch
            model.train()
        # Perform a final save.
        model.saver.save(session, os.path.join(opts.save_path, 'model.ckpt'),
                         global_step=model.global_step)


def make_feature_vectors():
    """Make feature vectors from a trained word2vec model for sentiment
    analysis"""
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            model = Word2Vec(opts, session, save_vocab=False)
        tf.train.import_meta_graph(opts.trained_model_meta)
        checkpoint_dir = os.path.dirname(opts.trained_model_meta)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        try:
            model.saver.restore(session, latest_checkpoint)
        except tf.errors.InvalidArgumentError as e:
            logging.error(e)
            logging.error('Make sure to supply the same training data as the'
                          ' model was trained with')
            sys.exit(1)
        model.make_averaged_feature_vectors(opts.reviews_training_path,
                                            opts.features_out_dir)
        model.make_averaged_feature_vectors(opts.reviews_validation_path,
                                            opts.features_out_dir)
        model.make_averaged_feature_vectors(opts.reviews_testing_path,
                                            opts.features_out_dir)


def make_interactive():
    """Start an interactive iPython session with a trained model"""
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            model = Word2Vec(opts, session, save_vocab=False)
        tf.train.import_meta_graph(opts.trained_model_meta)
        checkpoint_dir = os.path.dirname(opts.trained_model_meta)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        try:
            model.saver.restore(session, latest_checkpoint)
        except InvalidArgumentError as e:
            logging.error(e)
            logging.error('Make sure to supply the same training data as the'
                          ' model was trained with')
            sys.exit(1)
        # E.g.
        # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
        _start_shell(locals())


def main(_):
    if FLAGS.train_data and FLAGS.save_path and not FLAGS.trained_model_meta:
        run_training()
    elif FLAGS.train_data and FLAGS.trained_model_meta and FLAGS.interactive:
        make_interactive()
    elif (FLAGS.train_data and FLAGS.trained_model_meta and
          FLAGS.reviews_training_path and FLAGS.reviews_validation_path and
            FLAGS.reviews_testing_path and FLAGS.features_out_dir):
        make_feature_vectors()
    else:
        logging.info('To run training, specify --train_data and --save_path.\n'
                     'To start interactive shell specify --train_data, '
                     '--save_path and --interactive\n'
                     'To make feature vectors, specify --train_data,'
                     '--trained_model_meta, --reviews_training_path '
                     '--reviews_validation_path --reviews_testing_path '
                     '--features_out_dir')
        sys.exit(1)


if __name__ == "__main__":
    # This will automatically run main
    tf.app.run()
