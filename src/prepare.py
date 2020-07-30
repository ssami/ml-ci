import argparse
import os
import random


def _prep_data(train_path):
    file_list = os.listdir(train_path)
    file_contents = []
    for f in file_list:
        with open(os.path.join(train_path, f)) as fd:
            text = fd.read().strip()
            text = text.lower()
            file_contents.append(text)

    return file_contents


def get_samples(dir):
    pos_data = ['__label__pos ' + text for text in _prep_data(os.path.join(dir, 'pos'))]
    neg_data = ['__label__neg ' + text for text in _prep_data(os.path.join(dir, 'neg'))]

    return pos_data, neg_data


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('-x', dest='train_dir', help='input directory for train')
    argp.add_argument('-y', dest='test_dir', help='input directory for test')
    argp.add_argument('-p', dest='output_train', help='name of train file')
    argp.add_argument('-q', dest='output_test', help='name of test file')

    args = argp.parse_args()

    pos_train, neg_train = get_samples(args.train_dir)
    all_train = pos_train + neg_train
    random.shuffle(all_train)
    train_file = args.output_train
    with open(train_file, 'w') as fh:
        for line in all_train:
            fh.write(line + '\n')

    pos_test, neg_test = get_samples(args.test_dir)
    all_test = pos_test + neg_test
    random.shuffle(all_test)
    test_file = args.output_test
    with open(test_file, 'w') as fh:
        for line in all_test:
            fh.write(line + '\n')
