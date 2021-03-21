# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import argparse
import utils

argp = argparse.ArgumentParser()
argp.add_argument('--eval_corpus_path', help="Path of the corpus to evaluate on", default=None)
args = argp.parse_args()


def main():
    predictions = ['London'] * len(open(args.eval_corpus_path).readlines())
    total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
    else:
        print('Predictions written to {}; no targets provided'.format(args.outputs_path))


if __name__ == "__main__":
    main()