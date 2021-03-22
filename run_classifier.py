import argparse
import os

import pandas as pd

from classifier import FakeNewsClassifier


def main(args: argparse.Namespace) -> None:
    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    dev_df = pd.read_csv(os.path.join(args.data_dir, args.dev_file))
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))

    classifier = FakeNewsClassifier(use_cuda=args.use_cuda)
    if args.do_train:
        classifier.finetune(train_df=train_df, dev_df=dev_df,
                            evaluate_during_training=args.evaluate_during_training)
    if args.do_eval:
        results = classifier.evaluate(test_df=test_df)
        if args.disp_metrics:
            print(results)
        else:
            print(results[:results.find('\n')])

    if args.predict_headline:
        df = pd.DataFrame({'text_a': args.predict_headline, 'text_b': args.predict_body}, index=[0])
        print(classifier.predict(df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Data file containing training article data')
    parser.add_argument('--dev_file', type=str, default='dev.csv',
                        help='Data file containing validation article data')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Data file containing test article data')

    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Whether to evaluate the model on '
                             'validation data during training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')

    parser.add_argument('--predict_headline', type=str,
                        help='An article headline to make a single prediction from')
    parser.add_argument('--predict_body', type=str,
                        help='An article body to make a single prediction from')

    parser.add_argument('--disp_metrics', action='store_true',
                        help='Whether to display metrics (if set '
                             'to False, only F1 macro is displayed)')

    parser.add_argument('--use_cuda', action='store_true',
                        help='Whether to use GPU for training/evaluation')
    args = parser.parse_args()
    assert bool(args.predict_headline) == bool(args.predict_body), 'For making a single prediction both ' \
                                                                   '--predict_headline and --predict_body' \
                                                                   'are required'

    main(args)
