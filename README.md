# Word2Vec for Sentiment Analysis
## Udacity Machine Learning Engineer Nanodegree Capstone Project

The project presented here involves the automated prediction of movie review sentiments from IMDB based on the Word2vec word embedding model. Sentiment analysis is one of the fundamental tasks of natural language processing. It is commonly applied to process customer reviews, implement recommender systems and to understand sentiments in social media. [SemEval](https://en.wikipedia.org/wiki/SemEval) (short for semantic evaluation) is an annual competition designed to facilitate and track the improvements of semantic analysis systems. The best submissions to the SemEval 2016 Sentiment analysis in Twitter task overwhelmingly relied on word embeddings, the most popular of which was Word2vec. (Nakov et al., 2016)

Vector embeddings are learned vector representations of information such that the constructed vector space exhibits regularities useful for a specific task. Word embeddings are a natural language processing technique where a vector embedding is created from words or phrases. The purpose of these embeddings is to provide a representation of the input that is convenient to operate upon for other natural language processing or information retrieval methods. ([Wikipedia: Word embedding](https://en.wikipedia.org/wiki/Word_embedding)) The Word2vec model is popular because it offers state-of-the-art performance in evaluating semantic and syntactic similarities between words with relatively low training times. (Mikolov et al., 2013)

Read more and see references in the [Project Report.](report.pdf)

## Download data

All the data used and created during this project can be downloaded as a zip file from [here.](https://drive.google.com/file/d/0B3AvB1fXIbYTVURjSE44MlRjS0E/view?usp=sharing)

## Reproduce steps

The project can be reproduced by following the steps below.

### Install dependencies:
Install Python 2.7

Install pip
`pip install -r requirements.txt`

### Download raw dataset:
`mkdir -p data/raw`

Download raw dataset from [Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/data) to `data/raw`

### Create clean dataset:
```
mkdir -p data/clean
python -m sentiment_analysis.make_dataset \
    data/raw labeledTrainData.tsv \
    unlabeledTrainData.tsv \
    testData.tsv \
    data/clean \
    splitLabeledTrainData.tsv \
    splitLabeledValidationData.tsv
```

### Create bag of words feature vectors:

```
mkdir -p data/feature_vectors/bow
python -m sentiment_analysis.make_bag_of_words_features \
    data/clean/unlabeledTrainData.tsv \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/bow/splitTrainTermMatrix.pickle \
    data/feature_vectors/bow/splitValidationTermMatrix.pickle \
    data/feature_vectors/bow/testTermMatrix.pickle
```

### Create bag of words models:
```
mkdir -p data/predictions/bow
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/bow/splitTrainTermMatrix.pickle \
    data/feature_vectors/bow/splitValidationTermMatrix.pickle \
    data/feature_vectors/bow/testTermMatrix.pickle \
    data/predictions/bow
```

### Compile C++ TensorFlow ops:
```
cd sentiment_analysis/word2vec
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
```

Linux:

`g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0`

OS X:

`g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup`

`cd ../..`

### Train word2vec model:
```
mkdir -p data/word2vec
python -m sentiment_analysis.word2vec.word2vec \
    --train_data=data/clean/unlabeledTrainData.txt \
    --save_path=data/word2vec/
```

### Create feature vectors from word2vec model:
```
mkdir -p data/feature_vectors/word2vec/default
python -m sentiment_analysis.word2vec.word2vec \
    --trained_model_meta data/word2vec/default/model.ckpt-1591664.meta \
    --train_data data/clean/unlabeledTrainData.txt \
    --reviews_training_path data/clean/splitLabeledTrainData.tsv \
    --reviews_validation_path data/clean/splitLabeledValidationData.tsv \
    --reviews_testing_path data/clean/testData.tsv \
    --features_out_dir data/feature_vectors/word2vec/default
```

### Create word2vec models:
```
mkdir -p data/predictions/word2vec/default
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/word2vec/default/splitLabeledTrainDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/default/splitLabeledValidationDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/default/testDataFeatureVectors.pickle \
    data/predictions/word2vec/default
```

### Create word2vec models from clustered feature vectors:

Needs change to the source to work.

```
mkdir -p data/predictions/word2vec/default_clustered
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/word2vec/default_clustered/splitLabeledTrainDataClusteredFeatureVectors.pickle \
    data/feature_vectors/word2vec/default_clustered/splitLabeledValidationDataClusteredFeatureVectors.pickle \
    data/feature_vectors/word2vec/default_clustered/testDataClusteredFeatureVectors.pickle \
    data/predictions/word2vec/default_clustered &> data/predictions/word2vec/default_clustered/logs
```

### Create plots:
```
mkdir -p data/plots
python -m sentiment_analysis.visualize data/clean/unlabeledTrainData.tsv data/plots
```

### Create feature vectors from random embeddings:
```
mkdir -p data/feature_vectors/random_embedding:
python -m sentiment_analysis.make_random_embedding_feature_vectors \
    data/clean/unlabeledTrainData.tsv \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/random_embedding/splitTrainTermMatrix.pickle \
    data/feature_vectors/random_embedding/splitValidationTermMatrix.pickle \
    data/feature_vectors/random_embedding/testTermMatrix.pickle
```

### Create random embedding models:
```
mkdir -p data/predictions/random_embedding
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/random_embedding/splitTrainTermMatrix.pickle \
    data/feature_vectors/random_embedding/splitValidationTermMatrix.pickle \
    data/feature_vectors/random_embedding/testTermMatrix.pickle \
    data/predictions/random_embedding
```

### Train word2vec model without sub sampling:
```
mkdir -p data/word2vec/no_subsamp
python -m sentiment_analysis.word2vec.word2vec \
    --train_data=data/clean/unlabeledTrainData.txt \
    --save_path=data/word2vec/no_subsamp \
    --subsample=0
```

### Create feature vectors from word2vec no sub sampling:
```
mkdir -p data/feature_vectors/word2vec/no_subsamp
python -m sentiment_analysis.word2vec.word2vec \
    --trained_model_meta data/word2vec/no_subsamp/model.ckpt-2131034.meta \
    --train_data data/clean/unlabeledTrainData.txt \
    --reviews_training_path data/clean/splitLabeledTrainData.tsv \
    --reviews_validation_path data/clean/splitLabeledValidationData.tsv \
    --reviews_testing_path data/clean/testData.tsv \
    --features_out_dir data/feature_vectors/word2vec/no_subsamp
```

### Create word2vec no subsamp models:
```
mkdir -p data/predictions/word2vec/no_subsamp
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/word2vec/no_subsamp/splitLabeledTrainDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/no_subsamp/splitLabeledValidationDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/no_subsamp/testDataFeatureVectors.pickle \
    data/predictions/word2vec/no_subsamp
```

### Start interactive shell with word2vec model:
```
python -m sentiment_analysis.word2vec.word2vec \
    --trained_model_meta data/word2vec/default/model.ckpt-1591664.meta \
    --train_data data/clean/unlabeledTrainData.txt \
    --interactive=True
```

### Train word2vec with text8 corpus:
```
mkdir -p data/word2vec/text8
cd data/word2vec/text8
curl http://mattmahoney.net/dc/text8.zip > text8.zip
unzip text8.zip
rm text8.zip
cd ../../..
python -m sentiment_analysis.word2vec.word2vec \
    --train_data=data/word2vec/text8/text8 \
    --save_path=data/word2vec/text8
```

### Create feature vectors from word2vec trained on text8:
```
mkdir -p data/feature_vectors/word2vec/text8
python -m sentiment_analysis.word2vec.word2vec \
    --trained_model_meta data/word2vec/text8/model.ckpt-2264168.meta \
    --train_data data/word2vec/text8/text8 \
    --reviews_training_path data/clean/splitLabeledTrainData.tsv \
    --reviews_validation_path data/clean/splitLabeledValidationData.tsv \
    --reviews_testing_path data/clean/testData.tsv \
    --features_out_dir data/feature_vectors/word2vec/text8
```

### Evaluate feature vectors from word2vec model trained on text8 corpus:
```
mkdir -p data/predictions/word2vec/text8
python -m sentiment_analysis.models \
    data/clean/splitLabeledTrainData.tsv \
    data/clean/splitLabeledValidationData.tsv \
    data/clean/testData.tsv \
    data/feature_vectors/word2vec/text8/splitLabeledTrainDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/text8/splitLabeledValidationDataFeatureVectors.pickle \
    data/feature_vectors/word2vec/text8/testDataFeatureVectors.pickle \
    data/predictions/word2vec/text8 &> data/predictions/word2vec/text8/logs
```
