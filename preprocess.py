import argparse
import os
import random

import pandas as pd
from tqdm import tqdm


def split_data(df, seed=42):
    df_groups = list(df.groupby('id'))
    random.Random(seed).shuffle(df_groups)

    processed = 0
    dataset = {'train': [], 'dev': [], 'test': []}
    curr_set = 'test'

    test_size = 0.1 * len(df)
    dev_size = 0.1 * (len(df) - test_size)
    for group in tqdm(df_groups):
        if processed > test_size:
            curr_set = 'dev'
        if processed > test_size + dev_size:
            curr_set = 'train'

        for index, row in group[1].iterrows():
            dataset[curr_set].append(row)
            processed += 1

    train_data = pd.DataFrame(dataset['train'])
    dev_data = pd.DataFrame(dataset['dev'])
    test_data = pd.DataFrame(dataset['test'])

    return train_data, dev_data, test_data


def read_data(data_dir, bodies_file, stances_file):
    bodies = pd.read_csv(os.path.join(data_dir, bodies_file))
    stances = pd.read_csv(os.path.join(data_dir, stances_file))
    df = pd.merge(bodies, stances, on='Body ID', how='outer')
    df.columns = ['id', 'text_b', 'text_a', 'labels']

    return df


def main(args):
    df = read_data(args.data_dir, args.bodies_file, args.stances_file)
    train_data, dev_data, test_data = split_data(df, args.seed)
    train_data.to_csv(os.path.join(args.data_dir, args.train_file))
    dev_data.to_csv(os.path.join(args.data_dir, args.dev_file))
    test_data.to_csv(os.path.join(args.data_dir, args.test_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--bodies_file', type=str, default='bodies.csv',
                        help='Data file containing bodies of articles')
    parser.add_argument('--stances_file', type=str, default='stances.csv',
                        help='Data file containing stances of articles')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Data file to save training article data to')
    parser.add_argument('--dev_file', type=str, default='dev.csv',
                        help='Data file to save validation article data to')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Data file to save test article data to')
    parser.add_argument('--seed', type=str, default=42,
                        help='Random seed')
    args = parser.parse_args()
    main(args)
