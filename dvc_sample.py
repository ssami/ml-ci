# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import dvc.api
import boto3
import os
import random

import fasttext
# -

os.environ['AWS_ACCESS_KEY_ID'] = 'XXXX'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'XXXXXX'

# +
# needs to use the AWS credentials if the file is stored remotely! 

with dvc.api.open('data/data.xml', repo='https://github.com/ssami/ml-ci/') as fd: 
    for line in fd: 
        print(line)


# -

def prep_data(train_path): 
    file_list = os.listdir(train_path)
    file_contents = []
    for f in file_list: 
        with open(os.path.join(train_path, f)) as fd: 
            text = fd.read().strip()
            text = text.lower()
            file_contents.append(text)

    return file_contents


def get_training_samples(): 
    pos_data = ['__label__pos ' + text for text in 
                prep_data('/Users/ssami/projects/dvc-sample/movie_reviews/train/pos')]
    neg_data = ['__label__neg ' + text for text in 
                prep_data('/Users/ssami/projects/dvc-sample/movie_reviews/train/neg')]
    
    return pos_data, neg_data


def get_test_samples(): 
    pos_test = ['__label__pos ' + text for text in  
                prep_data('/Users/ssami/projects/dvc-sample/movie_reviews/test/pos')]
    neg_test = ['__label__neg ' + text for text in 
                prep_data('/Users/ssami/projects/dvc-sample/movie_reviews/test/neg')]
    
    return pos_test, neg_test


def train(): 
    pos_train, neg_train = get_training_samples()
    all_train = pos_train + neg_train
    random.shuffle(all_train)
    train_file = 'all_train.txt'
    with open(train_file, 'w') as fh: 
        for line in all_train: 
            fh.write(line + '\n')
    
    model = fasttext.train_supervised(train_file)
    
    return model


model = train()


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    f1 = 2 * (p*r)/(p+r)
    print("F1@{}\t{:.3f}".format(1, f1))


def test(model): 
    pos_test, neg_test = get_test_samples()
    all_test = pos_test + neg_test
    random.shuffle(all_test)
    test_file = 'all_test.txt'
    with open(test_file, 'w') as fh: 
        for line in all_test: 
            fh.write(line + '\n')
    
    print_results(*model.test(test_file))


test(model)


