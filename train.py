import argparse
import os
import random as python_random

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_TRACKING_URI = 'http://localhost:5000'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train a Keras classifier on the fetal health dataset '
                    'and log the run to MLflow.')

    data = parser.add_argument_group('data')
    data.add_argument('--data-path', default='fetal_health.csv',
                      help='CSV file to train on (default: %(default)s)')
    data.add_argument('--target', default='fetal_health',
                      help='Name of the label column (default: %(default)s)')
    data.add_argument('--test-size', type=float, default=0.2,
                      help='Fraction held out for testing '
                           '(default: %(default)s)')

    model = parser.add_argument_group('model')
    model.add_argument('--hidden', type=int, nargs='+', default=[32, 16],
                       metavar='N',
                       help='Units per hidden layer, e.g. --hidden 64 32 16 '
                            '(default: 32 16)')
    model.add_argument('--activation', default='relu',
                       help='Hidden layer activation (default: %(default)s)')
    model.add_argument('--optimizer', default='adam',
                       help='Optimizer (default: %(default)s)')

    training = parser.add_argument_group('training')
    training.add_argument('--epochs', type=int, default=50,
                          help='Training epochs (default: %(default)s)')
    training.add_argument('--batch-size', type=int, default=32,
                          help='Batch size (default: %(default)s)')
    training.add_argument('--validation-split', type=float, default=0.2,
                          help='Fraction of the training set used for '
                               'validation (default: %(default)s)')
    training.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: %(default)s)')
    training.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2],
                          help='Keras verbosity (default: %(default)s)')

    tracking = parser.add_argument_group('mlflow')
    tracking.add_argument(
        '--tracking-uri',
        default=os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_TRACKING_URI),
        help='MLflow tracking URI; falls back to $MLFLOW_TRACKING_URI '
             f'(default: {DEFAULT_TRACKING_URI})')
    tracking.add_argument('--experiment-name', default='fetal_health',
                          help='MLflow experiment (default: %(default)s)')
    tracking.add_argument('--run-name', default=None,
                          help='MLflow run name (default: auto-generated)')
    tracking.add_argument('--no-mlflow', action='store_true',
                          help='Train without logging to MLflow')

    args = parser.parse_args(argv)

    if not 0.0 < args.test_size < 1.0:
        parser.error('--test-size must be between 0 and 1')
    if not 0.0 <= args.validation_split < 1.0:
        parser.error('--validation-split must be in [0, 1)')
    if any(units < 1 for units in args.hidden):
        parser.error('--hidden units must be >= 1')

    return args


def reset_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path, target):
    data = pd.read_csv(path)
    if target not in data.columns:
        raise SystemExit(f'target column {target!r} not found in {path}')
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


def preprocess(X, y, test_size, seed):
    column_names = list(X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=column_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed)

    return (X_train.to_numpy(dtype='float32'),
            X_test.to_numpy(dtype='float32'),
            (y_train - 1).to_numpy(),
            (y_test - 1).to_numpy())


def build_model(n_features, n_classes, args):
    reset_seeds(args.seed)
    model = Sequential()
    model.add(InputLayer(shape=(n_features,)))
    for units in args.hidden:
        model.add(Dense(units, activation=args.activation))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=args.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(model, X_train, y_train, X_test, y_test, args):
    model.fit(X_train, y_train, epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=args.validation_split, verbose=args.verbose)
    return model.evaluate(X_test, y_test, verbose=0)


def main(argv=None):
    args = parse_args(argv)

    X, y = load_data(args.data_path, args.target)
    X_train, X_test, y_train, y_test = preprocess(
        X, y, args.test_size, args.seed)
    n_classes = len(np.unique(y_train))
    model = build_model(X.shape[1], n_classes, args)

    if args.no_mlflow:
        test_loss, test_accuracy = train(
            model, X_train, y_train, X_test, y_test, args)
    else:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow.tensorflow.autolog(log_models=True, log_input_examples=True,
                                  log_model_signatures=True, checkpoint=False)

        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_params({
                'cli_hidden': '-'.join(str(u) for u in args.hidden),
                'cli_activation': args.activation,
                'cli_seed': args.seed,
                'cli_test_size': args.test_size,
                'cli_data_path': args.data_path,
            })
            test_loss, test_accuracy = train(
                model, X_train, y_train, X_test, y_test, args)
            mlflow.log_metrics({'test_loss': test_loss,
                                'test_accuracy': test_accuracy})
        print(f'run_id: {run.info.run_id}')

    print(f'test_loss: {test_loss:.4f}')
    print(f'test_accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
