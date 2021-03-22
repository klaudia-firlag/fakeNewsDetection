from typing import List

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report

from metrics import calculate_f1_scores, get_conf_matrix_string, get_conf_matrix, fake_news_score, CLASSES

DEFAULT_ARGS = {
    'learning_rate': 3e-5,
    'num_train_epochs': 5,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'process_count': 10,
    'train_batch_size': 4,
    'eval_batch_size': 4,
    'max_seq_length': 512,
}


class FakeNewsClassifier:
    def __init__(self, model_type: str = 'bert', model_name: str = 'bert-base-uncased',
                 use_cuda: bool = False) -> None:
        self.model = ClassificationModel(model_type, model_name, num_labels=4,
                                         args=DEFAULT_ARGS, use_cuda=use_cuda)

    def finetune(self, train_df: pd.DataFrame, dev_df: pd.DataFrame = None,
                 evaluate_during_training: bool = False) -> None:

        train_df['labels'] = list(map(lambda x: CLASSES[x], train_df['labels']))

        if evaluate_during_training:
            self.model.train_model(train_df, eval_df=dev_df, multi_label=True, show_running_loss=True)
        else:
            self.model.train_model(train_df, multi_label=True, show_running_loss=True)

    def evaluate(self, test_df: pd.DataFrame) -> str:
        test_df['labels'] = list(map(lambda x: CLASSES[x], test_df['labels']))

        _, model_outputs_test, _ = self.model.eval_model(test_df)
        preds_test = np.argmax(model_outputs_test, axis=1)

        result_string = calculate_f1_scores(preds_test, test_df['labels'])

        conf_matrix = get_conf_matrix(preds_test, test_df['labels'])
        fnc_score = fake_news_score(preds_test, test_df['labels'])
        result_string += f'\nRelative FNC Score: {100 / 13204.75 * fnc_score:.3f}%\n'
        result_string += get_conf_matrix_string(conf_matrix)

        eval_report = classification_report(test_df['labels'], preds_test)
        result_string += 'Test report'
        result_string += eval_report

        return result_string

    def predict(self, test_df: pd.DataFrame) -> List[str]:
        def get_label_name(label_int):
            return list(CLASSES.keys())[list(CLASSES.values()).index(label_int)]

        _, model_outputs_test, _ = self.model.eval_model(test_df)
        preds_test = np.argmax(model_outputs_test, axis=1)
        return get_label_name(preds_test[0])
