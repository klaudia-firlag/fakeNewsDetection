import unittest

import pandas as pd

from classifier import FakeNewsClassifier
from metrics import CLASSES
from preprocess import read_data, split_data

BODY = """Judd Nelson rebuffs Internet rumors that he died of a drug overdose. Rumors swirled on 
Twitter that the ""Breakfast Club"" actor had found dead in his apartment of an apparent drug 
overdose. His agent Gregg Klein tweeted out a photo of Nelson holding today's paper"""
HEADLINE = 'Parliament Hill shooting: Sergeant-At-Arms Kevin Vickers the family hero who took down Ottawa gunman'
LABEL = CLASSES['unrelated']
DATAFRAME = pd.DataFrame({'text_a': HEADLINE, 'text_b': BODY, 'labels': LABEL}, index=[0])


class TestClassifier(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestClassifier, self).__init__(*args, **kwargs)
        self.data_dir = 'data'
        self.bodies_file = 'bodies.csv'
        self.stances_file = 'stances.csv'

        self.df_columns = ['id', 'text_b', 'text_a', 'labels']

        self.classifier = FakeNewsClassifier()

    def assert_valid_df(self, df):
        self.assertIsInstance(df, pd.DataFrame)
        for col in self.df_columns:
            self.assertIn(col, df.columns)

    def test_read_data(self):
        df = read_data(self.data_dir, self.bodies_file, self.stances_file)

        self.assert_valid_df(df)
        self.assertIsInstance(df.iloc[0]['text_a'], str)
        self.assertIsInstance(df.iloc[0]['text_b'], str)
        self.assertIn(df.iloc[0]['labels'], CLASSES.keys())

    def test_split_data(self):
        df = read_data(self.data_dir, self.bodies_file, self.stances_file)
        train_data, dev_data, test_data = split_data(df)

        for data in train_data, dev_data, test_data:
            self.assert_valid_df(data)

        self.assertLess(len(train_data), 0.9 * len(df))
        self.assertLess(len(dev_data), 0.1 * len(df))
        self.assertAlmostEqual(len(test_data), 0.1 * len(df), delta=100)

    def test_single_prediction(self):
        predictions = self.classifier.predict(DATAFRAME)
        self.assertIsInstance(predictions, str)
        self.assertEqual(predictions, 'unrelated')


if __name__ == '__main__':
    unittest.main()
