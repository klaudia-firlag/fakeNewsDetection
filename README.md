# Fake News Detection

## Input
A headline and a body text - either from the same news article or from two different articles.

## Output
Classify the stance of the body text relative to the claim made in the headline into one of four categories:
   * Agrees: The body text agrees with the headline.
   * Disagrees: The body text disagrees with the headline.
   * Discusses: The body text discuss the same topic as the headline, but does not take a position
   * Unrelated: The body text discusses a different topic than the headline

## Oficial Scoring
![alt text](http://www.fakenewschallenge.org/assets/img/fnc-eval.png)

## Tests
```shell
python -m unittest unittests.py
```

## Run
### Preprocess data
Arguments:
* `--data_dir` - a directory containing the original data files - default: `data`
* `--bodies_file` - data file containing bodies of articles - default: `bodies.csv`
* `--stances_file` - data file containing stances of articles - default: `stances.csv`
* `--train_file` - data file to save training article data to - default: `train.csv`
* `--dev_file` - data file to save validation article data to - default: `dev.csv`
* `--test_file` - data file to save test article data to - default: `test.csv`
* `--seed` - random seed - default: `42`

Example run:
```shell
python preprocess.py
```
### Run training and/or evaluation and/or a single prediction
Arguments:
* `--data_dir` - a directory containing the original data files - default: `data`
* `--train_file` - data file containing training article data - default: `train.csv`
* `--dev_file` - data file containing validation article data - default: `dev.csv`
* `--test_file` - data file containing test article data - default: `test.csv`
* `--do_train` - a flag for training the model
* `--evaluate_during_training` - a flag for evaluating the model on validation data during training
* `--do_eval` - a flag for evaluating the model
* `--predict_headline` - an article headline to make a single prediction from (must be used with `--predict_body`)
* `--predict_body` - an article body to make a single prediction from (must be used with `--predict_headline`)
* `--disp_metrics` - a flag for displaying metrics (if set to False, only F1 macro is displayed)

Example run:
```shell
python run_classifier.py --do_train --evaluate_during_training --do_eval --disp_metrics
```
