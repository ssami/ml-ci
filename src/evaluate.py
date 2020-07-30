import argparse

import fasttext


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    f1 = 2 * (p * r) / (p + r)
    print("F1@{}\t{:.3f}".format(1, f1))


def test(model_loc, test_file):
    model = fasttext.load_model(model_loc)
    print_results(*model.test(test_file))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('-m', dest='model', help='input directory for train')
    argp.add_argument('-t', dest='test_file', help='input directory for test')

    args = argp.parse_args()

    test(args.model, args.test_file)
