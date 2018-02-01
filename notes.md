Labeleled dataset postivie/negative label ratio: 1.0 (12500 of each)

using LogisticRegressionCV for cross-validated logistic regression
use grid search for SVC as well?

Logistic Regression CV. Evaluation ROC AUC score: 0.938364933839. Testing ROC AUC score: 0.86872. Difference is that latter only uses a single discrimination threshold of presumably 0.5. TODO: try different discrimination thresholds?

Check out [Adversarial Training Methods for Semi-Supervised Text Classification](https://github.com/tensorflow/models/tree/master/adversarial_text)

word2vec not good, bc antonyms are close to each other

weighing by tf-idf is not good, bc important words like "good" or "bad" have larger frequencies

use text summarization techniques to compress input?

extract adjectives, verbs from text

issues: adjectives, verbs might describe plot which is neutral

sum vs mean only makes sense if there is a correlation between the length of the vector and the label.

correlation between word count and sentiment is 0.016 using `data/clean/labeledTrainData.tsv`

word2vec mean feature vectors Kaggle eval: 0.82040 

see how random vector for each word performs.

often only small part of review is relevant for the sentiment as longer parts usually describe the plot

Total words in clean unlabeled dataset incl stop words: 11,876,820

Is there an overlap between the reviews where bow fails vs where Word2vec fails?

Plot of ROC

Remove infrequent words

LogisticRegressionCV random embeddings Kaggle eval: 0.72428

LogisticRegressionCV default Word2vec mean feature vectors C param: 166.81005372, intercept: -1.82824357

Word2vec default clustered logistic regression cv testing eval: 0.80712

SVC word2vec default clustered testing eval: 0.80948

[2017-07-02 18:51:07,472 - INFO:word2vec.py:nearby:369]
good
=====================================
[2017-07-02 18:51:07,472 - INFO:word2vec.py:nearby:371] good                 1.0000
[2017-07-02 18:51:07,472 - INFO:word2vec.py:nearby:371] decent               0.6669
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] nice                 0.6342
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] great                0.6309
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] fine                 0.6209
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] bad                  0.6160
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] brief                0.5473
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] excellent            0.5187
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] ok                   0.5182
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] fantastic            0.5168
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:369]
bad
=====================================
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] bad                  1.0000
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] stupid               0.6430
[2017-07-02 18:51:07,473 - INFO:word2vec.py:nearby:371] terrible             0.6168
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] good                 0.6160
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] horrible             0.6152
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] keep                 0.5585
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] georg                0.5499
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] lame                 0.5467
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] cruel                0.5425
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] ridiculous           0.5376
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:369]
terrible
=====================================
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] terrible             1.0000
[2017-07-02 18:51:07,474 - INFO:word2vec.py:nearby:371] horrible             0.8230
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] keep                 0.7191
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] wanna                0.7073
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] humorous             0.6668
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] scientific           0.6564
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] ridiculous           0.6215
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] poor                 0.6212
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] cruel                0.6188
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] cameras              0.6169
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:369]
excellent
=====================================
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] excellent            1.0000
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] soldiers             0.8008
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] brief                0.6989
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] amazing              0.6823
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] former               0.6822
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] fantastic            0.6807
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] wonderful            0.6794
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] fourth               0.6769
[2017-07-02 18:51:07,475 - INFO:word2vec.py:nearby:371] brilliant            0.6449
[2017-07-02 18:51:07,476 - INFO:word2vec.py:nearby:371] recommended          0.6372
