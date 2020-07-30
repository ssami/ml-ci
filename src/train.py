import argparse

import fasttext


def train(train_file, model_save_path, validation=None):
    if validation:
        model = fasttext.train_supervised(train_file, autotuneValidationFile=validation)
    else:
        model = fasttext.train_supervised(train_file)

    model.save_model(model_save_path)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('-f', dest='train_data', help='full labeled training file')
    argp.add_argument('-g', dest='model_dest', help='model store path ')
    argp.add_argument('-v', dest='validation', default=None, help='optionally auto-tune with validation file')

    args = argp.parse_args()

    train(args.train_data, args.model_dest, args.validation)



