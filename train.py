import argparse
import os
import random as python_random
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from keras.wrappers import SKLearnClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TRACKING_URI = ('https://dagshub.com/pederzinidaniel/'
                        'my-first-repo.mlflow')
DOTENV_PATH = Path(__file__).with_name('.env')


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
    tracking.add_argument('--registered-model-name',
                          default='fetal_health_classifier',
                          help='Name to register the model under in the '
                               'MLflow Model Registry '
                               '(default: %(default)s)')
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
    """Split into train/test. Scaling is handled by the pipeline so that
    the scaler is fit on the training split only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    return (X_train.to_numpy(dtype='float32'),
            X_test.to_numpy(dtype='float32'),
            (y_train - 1).to_numpy(),
            (y_test - 1).to_numpy())


def build_model(n_features, n_classes, args,
                loss='sparse_categorical_crossentropy'):
    reset_seeds(args.seed)
    model = Sequential()
    model.add(InputLayer(shape=(n_features,)))
    for units in args.hidden:
        model.add(Dense(units, activation=args.activation))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=args.optimizer, loss=loss,
                  metrics=['accuracy'])
    return model


def _build_wrapped_model(X, y, args):
    """Model factory for SKLearnClassifier, which one-hot encodes the
    target before calling this and passes X/y as keyword arguments."""
    n_classes = y.shape[1] if y.ndim > 1 else len(np.unique(y))
    return build_model(X.shape[1], n_classes, args,
                       loss='categorical_crossentropy')


def build_pipeline(args):
    """Scaler + Keras classifier as one estimator, so that inference
    applies exactly the same scaling that was fit during training."""
    classifier = SKLearnClassifier(
        model=_build_wrapped_model,
        model_kwargs={'args': args},
        fit_kwargs={'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'validation_split': args.validation_split,
                    'verbose': args.verbose})
    return Pipeline([('scaler', StandardScaler()),
                     ('classifier', classifier)])


def train(pipeline, X_train, y_train, X_test, y_test, args):
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.decision_function(X_test)
    test_loss = log_loss(y_test, probabilities,
                         labels=np.unique(y_train))
    test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return test_loss, test_accuracy


def log_pipeline(pipeline, input_example, run_id, args):
    """Save, upload, and register the fitted pipeline for a run.

    cloudpickle is required because the default skops format cannot
    serialize the Keras network inside the pipeline. Saving and uploading
    the artifacts separately avoids MLflow 3's logged-model metric backfill,
    which DagsHub rejects with BAD_REQUEST.
    """
    with tempfile.TemporaryDirectory(prefix='mlflow-model-') as local_dir:
        mlflow.sklearn.save_model(
            pipeline, local_dir, input_example=input_example,
            serialization_format='cloudpickle')
        mlflow.log_artifacts(local_dir, artifact_path='model')

    return mlflow.register_model(
        f'runs:/{run_id}/model', args.registered_model_name)


def main(argv=None):
    load_dotenv(DOTENV_PATH)
    args = parse_args(argv)

    X, y = load_data(args.data_path, args.target)
    X_train, X_test, y_train, y_test = preprocess(
        X, y, args.test_size, args.seed)
    pipeline = build_pipeline(args)

    if args.no_mlflow:
        test_loss, test_accuracy = train(
            pipeline, X_train, y_train, X_test, y_test, args)
    else:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow.tensorflow.autolog(log_models=False, log_input_examples=True,
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
                pipeline, X_train, y_train, X_test, y_test, args)
            mlflow.log_metrics({'test_loss': test_loss,
                                'test_accuracy': test_accuracy})
            log_pipeline(pipeline, X_test[:2], run.info.run_id, args)
        print(f'run_id: {run.info.run_id}')

    print(f'test_loss: {test_loss:.4f}')
    print(f'test_accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
