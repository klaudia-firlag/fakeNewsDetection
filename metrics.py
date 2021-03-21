from typing import List

from numpy.core._multiarray_umath import ndarray
from sklearn.metrics import f1_score

CLASSES = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


def calculate_f1_scores(y_true, y_predicted) -> str:
    f1_macro = f1_score(y_true, y_predicted, average='macro')
    f1_classwise = f1_score(y_true, y_predicted, average=None, labels=list(CLASSES.keys()))

    return f'F1 macro: {f1_macro * 100:.3f}%\n' \
           f'F1 agree: {f1_classwise[0] * 100:.3f}%\n' \
           f'F1 disagree: {f1_classwise[1] * 100:.3f}%\n' \
           f'F1 discuss: {f1_classwise[2] * 100:.3f}%\n' \
           f'F1 unrelated: {f1_classwise[3] * 100:.3f}'


def fake_news_score(predictions, gold_labels) -> float:
    score = 0.0
    for i, (pred, gold) in enumerate(zip(predictions, gold_labels)):
        if pred == gold:
            score += 0.25
            if pred != 3:
                score += 0.50
        if pred != 3 and gold != 3:
            score += 0.25
    return score


def get_conf_matrix(predictions: ndarray, gold_labels: List[int]) -> List[List[int]]:
    conf_matrix = [[0] * 4] * 4
    for i, (pred, gold) in enumerate(zip(predictions, gold_labels)):
        conf_matrix[pred][gold] += 1
    return conf_matrix


def get_conf_matrix_string(conf_matrix) -> str:
    lines = ['CONFUSION MATRIX:']
    header = '|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|'.format('', *list(CLASSES.keys()))
    line_len = len(header)
    lines.append('-' * line_len)
    lines.append(header)
    lines.append('-' * line_len)
    hit = 0
    total = 0
    for i, row in enumerate(conf_matrix):
        hit += row[i]
        total += sum(row)
        lines.append('|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|'.format(list(CLASSES.keys())[i], *row))
        lines.append('-' * line_len)
    lines.append('ACCURACY: {(hit / total) * 100:.3f}%')
    return '\n'.join(lines)
