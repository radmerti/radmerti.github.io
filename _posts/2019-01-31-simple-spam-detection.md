---
layout: post
date: 2019-01-31 11:24:38
tags: [python, notebook]
title: "Simple Spam Filter Using Back-of-Words"
summary: >
  I use a simple back-of-words approach and a multinomial Naive Bayes classifier
  to detect spam mails.
row_span: 1
---

This is a script that trains a Multinomial Naive Bayes model to detect spam mails. The script can be executed as follows:

To train a model run

```
script.py train path/to/spam path/to/nospam
```

To evaluate the model using cross-validation (using
[cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html))
run

```
script.py cross path/to/spam path/to/nospam
```

To find the best set of parameters using a grid search (using
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html))
run

```
script.py grid path/to/spam path/to/nospam
```

The grid search does a limited sweap over the vocabulary size (no limit, 1000000)
and the percentage used for detecting stop words (1.0, 0.99, 0.98, 0.95). The
latter parameter drops a word from the vocabulary if it occures in more than x
percent of the documents.


# Spam Detection

## The Problem

Classifying e-mails into spam and no-spam is not an easy task because the
distribution of labels is very inbalanced. Often well over 95% of the labels
are spam mails. This makes training a classifier diffictult because the naive
prediction (always predicting spam) already achieves 95% accuracy. Of course,
this solution is useless as all relevant mails will be filtered out as well.

There are several strategies to overcome this problem:

* oversampling the minority class to get a 50/50 distribution of labels
* undersampling the majority class to get a 50/50 distribution of labels
* generating new samples in the minority class that are close to the existing
  samples in the feature space

In this script I didn't employ any of these techniques as the dataset was
already balanced.

## Simple NLP Approach

Many features from the mails can be used to classify spam. For example one can
use meta information from the mail header like time, IP address, sender and so
forth. I'm sure suffisticated spam filter systems use this kind of information,
but modern spam filters all rely heavily on natural language processing to use
the mail body for classification. This is the approach that is demonstrated
here.

### Bag of Words

In this script I use the approcha of modelling the natural language as a
bag-of-words. That is for every mail we have a long vector that is the length
of the vocabulary were each entry represents the number of times that word
occures in the mail. In scikiti-learn this can be done with the
[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

Additionally one approach that is often used in document classification and
retrieval is to use the so called TF-IDF statistic. It stands for term
[term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
The term frequency of a term of a specific document is weighted by teh inverse
document frequency of that term over the whole corpus.

The  The idea is that some words have a high frequency in a document but also
occure in a lot of documents in the corpus. They say less about the specific
document compared to a word that is frequent in the document but doesn't occure
in a lot of documents.

Here I use the [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
from scikit-learn to generate the TF-IDF feature out of the word count matrix
(word count for each word in the vocabulary for all documents).

### Naive Bayes

For classification I use a Naive Bayes based on the multinomial distribution
which models the probability of counts. Perfect! Specifically, I use the
[MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
model from scikit-learn.


```python
import sys
import os
import pickle
import numpy
from pandas import DataFrame
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# TODO: SVM Model
# TODO: Try Binary feature
# TODO: subject line feature

# TODO: Grid search for Binary vs TF vs TF*IDF


def load_from_folders(dirs):
    print(dirs)
    data = DataFrame({'body': [], 'label': []})
    for path, label in dirs:
        mails, fnames = [], []
        for root_dir, dir_names, file_names in os.walk(path):
            # load data in sub-directories
            for directory in dir_names:
                data = data.append(load_from_folders([(os.path.join('.',directory),label)]))
            # load files in root directory
            for file_name in file_names:
                file_path = os.path.join(root_dir, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == '\n':
                            past_header = True
                    f.close()
                    content = '\n'.join(lines)
                    mails.append({'body': content, 'label': label})
                    fnames.append(file_path)
        data = data.append(DataFrame(mails, index=fnames))
    return data

def create_dataframe(dirs):
    data = load_from_folders(dirs)
    #data.reset_index().drop_duplicates(subset='index').set_index('index')
    return data.reindex(numpy.random.permutation(data.index))

def write_prediction(prediction, file_name):
    f = open(file_name, "w", encoding="latin-1")

    for line in zip(prediction.index.values, prediction['label']):
        f.write('{0}\t{1}\n'.format(line[0].split('/')[-1], line[1]))
        #print('{0}\t{1}'.format(line[0], line[1]))

    f.close()

def create_model():
    # HashingVectorizer ?
    return Pipeline([
            ('count_vectorizer', CountVectorizer(
                ngram_range=(1, 2), 
                strip_accents='unicode', 
                min_df=2, 
                max_df=0.90, 
                stop_words=None, 
                max_features=1000000,
                binary=True)),
            # ('idf_transformer', TfidfTransformer(
            # norm='l2', 
            # use_idf=True, 
            # smooth_idf=True, 
            # sublinear_tf=False)),
            ('classifier', MultinomialNB(
                alpha=0.001, 
                fit_prior=True, 
                class_prior=None))
        ])

def load_model(model_name):
    with open(model_name, 'rb') as f:
        model_attributes = pickle.load(f)
        pipeline = create_model()
        pipeline.named_steps['count_vectorizer'].vocabulary_ = model_attributes[0]
        pipeline.named_steps['count_vectorizer'].stop_words_ = None

        pipeline.named_steps['classifier'].class_count_ = model_attributes[1]
        pipeline.named_steps['classifier'].feature_count_ = model_attributes[2]
        pipeline.named_steps['classifier'].class_log_prior_ = numpy.log(numpy.divide(
            model_attributes[1],
            numpy.sum(model_attributes[1])
        ))
        pipeline.named_steps['classifier'].feature_log_prob_ = numpy.transpose(numpy.log(numpy.multiply(
            numpy.transpose(pipeline.named_steps['classifier'].feature_count_),
            numpy.divide(1.0, pipeline.named_steps['classifier'].class_count_)
        )))
        pipeline.named_steps['classifier'].classes_ = model_attributes[3]

        return pipeline

def save_pipeline(pipeline, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump([
            pipeline.named_steps['count_vectorizer'].vocabulary_, 
            pipeline.named_steps['classifier'].class_count_, 
            pipeline.named_steps['classifier'].feature_count_,
            pipeline.named_steps['classifier'].classes_], f)

if __name__ == "__main__":
    arguments = sys.argv

    if len(sys.argv) == 5:
        if arguments[1] == 'classify':
            # load model
            pipeline = load_model(arguments[2])

            # load mails from directory
            data = create_dataframe([(arguments[3], '')])

            # predict class labels
            data['label'] = pipeline.predict(data['body'])

            # output the result
            print('\nTotal emails classified:', len(data), 
                '\nvocab size:', len(pipeline.named_steps['count_vectorizer'].vocabulary_))

            write_prediction(data, arguments[4])

        elif arguments[1] == 'learn':
            # load training data
            data = create_dataframe([(arguments[2], 'SPAM'), (arguments[3], 'NOSPAM')])

            # create pipeline
            pipeline = create_model()

            # train classifier
            pipeline.fit(data['body'].values, data['label'].values.astype(str))

            # save the model
            save_pipeline(pipeline, arguments[4])

            with open('backup.model', 'wb') as f:
                pickle.dump(pipeline, f)

        elif arguments[1] == 'cross':
            # load training data
            data = create_dataframe([(arguments[2], 'SPAM'), (arguments[3], 'NOSPAM')])

            # create pipeline
            pipeline = create_model()

            # perform 10-fold crossvalidation
            scores = cross_val_score(pipeline, data['body'].values, data['label'].values.astype(str), cv=10, n_jobs=2, pre_dispatch=3)

            # train classifier
            pipeline.fit(data['body'].values, data['label'].values.astype(str))

            # output the result
            print('Total emails classified:', len(data), '\nvocab size:', len(pipeline.named_steps['count_vectorizer'].vocabulary_), 
                '\nstop words:', len(pipeline.named_steps['count_vectorizer'].stop_words_), '\n\n10-Fold-Cross-Validation:')
            for index, kscore in enumerate(scores):
                print('{0:3}: {1:.3f}  '.format(index+1, kscore), end='\n')
            print('----------\nAvg: {0:.3f}'.format(sum(scores)/len(scores)))

            # save the model
            save_pipeline(pipeline, arguments[4])

        elif arguments[1] == 'grid':
            # load training data
            data = create_dataframe([(arguments[2], 'SPAM'), (arguments[3], 'NOSPAM')])

            # create pipeline
            pipeline = create_model()

            # define parameters for grid
            param_grid = dict(count_vectorizer__max_features=[None, 1000000],
                              count_vectorizer__max_df=[1.0, 0.99, 0.98, 0.95])
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=0, cv=5, n_jobs=2, pre_dispatch=2)

            # start grid search
            grid_search.fit(data['body'].values, data['label'].values.astype(str))

            # print result
            print('Cross-Validation result:\n', grid_search.grid_scores_)
            print('\nbest\n', grid_search.best_params_, '\nscore:', grid_search.best_score_)

            # save the best model
            save_pipeline(pipeline, arguments[4])

        else:
            print('Mode', arguments[1], 'not known.')
    else:
        print('Wrong number of command line arguments!')
```
