import argparse
import json

import fasttext


def _print_results(N, p, r, metrics_file):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    f1 = 2 * (p * r) / (p + r)
    print("F1@{}\t{:.3f}".format(1, f1))

    metrics = {
        'p': p,
        'r': r,
        'f1': f1
    }

    with open(metrics_file, 'w') as fh:
        fh.write(json.dumps(metrics))


def test(model_loc, test_file, metrics_file):
    model = fasttext.load_model(model_loc)
    _print_results(*model.test(test_file), metrics_file)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('-m', dest='model', help='input directory for train')
    argp.add_argument('-t', dest='test_file', help='input directory for test')
    argp.add_argument('-j', dest='metrics_file', help='name of scores file')

    args = argp.parse_args()

    test(args.model, args.test_file, args.metrics_file)
