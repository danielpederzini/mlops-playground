import contextlib
import types

import mlflow.tensorflow
import numpy as np
import pandas as pd
import pytest

import train as train_module

FEATURES = ['a', 'b', 'c', 'd']


@pytest.fixture
def csv_path(tmp_path):
    """A small well-formed dataset with 1-based labels, like the real one."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((60, len(FEATURES))), columns=FEATURES)
    frame['fetal_health'] = np.tile([1.0, 2.0, 3.0], 20)
    path = tmp_path / 'sample.csv'
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fast_args(csv_path):
    return ['--data-path', csv_path, '--epochs', '1', '--hidden', '4',
            '--verbose', '0']


class TestParseArgs:
    def test_defaults_match_documented_behaviour(self):
        args = train_module.parse_args([])
        assert args.data_path == 'fetal_health.csv'
        assert args.target == 'fetal_health'
        assert args.epochs == 50
        assert args.batch_size == 32
        assert args.hidden == [32, 16]
        assert args.seed == 42
        assert args.test_size == 0.2
        assert args.validation_split == 0.2
        assert args.no_mlflow is False

    def test_overrides(self):
        args = train_module.parse_args([
            '--epochs', '7', '--batch-size', '8', '--hidden', '64', '32', '16',
            '--seed', '3', '--activation', 'tanh', '--optimizer', 'sgd',
            '--run-name', 'demo', '--no-mlflow'])
        assert args.epochs == 7
        assert args.batch_size == 8
        assert args.hidden == [64, 32, 16]
        assert args.seed == 3
        assert args.activation == 'tanh'
        assert args.optimizer == 'sgd'
        assert args.run_name == 'demo'
        assert args.no_mlflow is True

    def test_verbose_default_is_a_valid_keras_value(self):
        default = train_module.parse_args([]).verbose
        assert train_module.parse_args(
            ['--verbose', str(default)]).verbose == default

    def test_tracking_uri_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://example:1234')
        assert train_module.parse_args([]).tracking_uri == \
            'http://example:1234'

    def test_tracking_uri_default_without_env(self, monkeypatch):
        monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
        assert train_module.parse_args([]).tracking_uri == \
            train_module.DEFAULT_TRACKING_URI

    @pytest.mark.parametrize('argv', [
        ['--test-size', '1.5'],
        ['--test-size', '0'],
        ['--validation-split', '1.0'],
        ['--validation-split', '-0.1'],
        ['--hidden', '8', '0'],
        ['--hidden', '-4'],
        ['--verbose', '3'],
        ['--epochs', 'many'],
    ])
    def test_invalid_values_are_rejected(self, argv):
        with pytest.raises(SystemExit):
            train_module.parse_args(argv)


class TestLoadData:
    def test_splits_features_from_target(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        assert list(X.columns) == FEATURES
        assert 'fetal_health' not in X.columns
        assert len(X) == len(y) == 60

    def test_missing_target_exits_with_message(self, csv_path):
        with pytest.raises(SystemExit) as excinfo:
            train_module.load_data(csv_path, 'nope')
        assert 'nope' in str(excinfo.value)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            train_module.load_data(str(tmp_path / 'absent.csv'), 'x')


class TestPreprocess:
    def test_shapes_and_split_ratio(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        X_train, X_test, y_train, y_test = train_module.preprocess(
            X, y, test_size=0.2, seed=42)
        assert X_train.shape == (48, 4)
        assert X_test.shape == (12, 4)
        assert y_train.shape == (48,)
        assert y_test.shape == (12,)

    def test_returns_float32_arrays_for_keras(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        X_train, X_test, _, _ = train_module.preprocess(X, y, 0.2, 42)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32

    def test_labels_are_shifted_to_zero_based(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        _, _, y_train, y_test = train_module.preprocess(X, y, 0.2, 42)
        assert set(np.unique(y_train)) <= {0, 1, 2}
        assert set(np.unique(y_test)) <= {0, 1, 2}
        assert y_train.min() >= 0

    def test_features_are_standardised(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        X_train, X_test, _, _ = train_module.preprocess(X, y, 0.2, 42)
        combined = np.vstack([X_train, X_test])
        assert np.allclose(combined.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(combined.std(axis=0), 1, atol=1e-5)

    def test_split_is_deterministic_for_a_seed(self, csv_path):
        X, y = train_module.load_data(csv_path, 'fetal_health')
        first = train_module.preprocess(X, y, 0.2, 42)[0]
        second = train_module.preprocess(X, y, 0.2, 42)[0]
        different = train_module.preprocess(X, y, 0.2, 7)[0]
        assert np.array_equal(first, second)
        assert not np.array_equal(first, different)


class TestBuildModel:
    def test_layer_count_follows_hidden_argument(self):
        args = train_module.parse_args(['--hidden', '8', '4', '2'])
        model = train_module.build_model(4, 3, args)
        assert len(model.layers) == 4

    def test_input_and_output_shapes(self):
        args = train_module.parse_args(['--hidden', '8'])
        model = train_module.build_model(4, 3, args)
        assert model.input_shape == (None, 4)
        assert model.output_shape == (None, 3)

    def test_output_units_track_n_classes(self):
        args = train_module.parse_args(['--hidden', '8'])
        assert train_module.build_model(4, 5, args).output_shape == (None, 5)

    def test_same_seed_gives_same_initial_weights(self):
        args = train_module.parse_args(['--hidden', '8', '--seed', '42'])
        first = train_module.build_model(4, 3, args).get_weights()[0]
        second = train_module.build_model(4, 3, args).get_weights()[0]
        assert np.array_equal(first, second)

    def test_activation_and_optimizer_are_applied(self):
        args = train_module.parse_args(
            ['--hidden', '8', '--activation', 'tanh', '--optimizer', 'sgd'])
        model = train_module.build_model(4, 3, args)
        assert model.layers[0].activation.__name__ == 'tanh'
        assert model.optimizer.name.lower() == 'sgd'


class TestMain:
    def test_no_mlflow_run_trains_without_touching_mlflow(
            self, fast_args, monkeypatch, capsys):
        def explode(*args, **kwargs):
            raise AssertionError('mlflow must not be used with --no-mlflow')

        monkeypatch.setattr(train_module.mlflow, 'set_tracking_uri', explode)
        monkeypatch.setattr(train_module.mlflow, 'set_experiment', explode)
        monkeypatch.setattr(train_module.mlflow, 'start_run', explode)

        train_module.main(fast_args + ['--no-mlflow'])

        out = capsys.readouterr().out
        assert 'test_accuracy:' in out
        assert 'test_loss:' in out
        assert 'run_id:' not in out

    def test_mlflow_path_logs_expected_params_and_metrics(
            self, fast_args, monkeypatch, capsys):
        calls = {'params': {}, 'metrics': {}, 'autolog': {}}

        @contextlib.contextmanager
        def fake_start_run(run_name=None, **kwargs):
            calls['run_name'] = run_name
            yield types.SimpleNamespace(
                info=types.SimpleNamespace(run_id='abc123'))

        monkeypatch.setattr(train_module.mlflow, 'set_tracking_uri',
                            lambda uri: calls.__setitem__('uri', uri))
        monkeypatch.setattr(train_module.mlflow, 'set_experiment',
                            lambda name: calls.__setitem__('experiment', name))
        monkeypatch.setattr(mlflow.tensorflow, 'autolog',
                            lambda **kw: calls['autolog'].update(kw))
        monkeypatch.setattr(train_module.mlflow, 'start_run', fake_start_run)
        monkeypatch.setattr(train_module.mlflow, 'log_params',
                            calls['params'].update)
        monkeypatch.setattr(train_module.mlflow, 'log_metrics',
                            calls['metrics'].update)

        train_module.main(fast_args + ['--experiment-name', 'exp',
                                       '--tracking-uri', 'http://x:1',
                                       '--run-name', 'named'])

        assert calls['uri'] == 'http://x:1'
        assert calls['experiment'] == 'exp'
        assert calls['run_name'] == 'named'
        assert calls['autolog']['checkpoint'] is False
        assert calls['autolog']['log_models'] is True
        assert calls['params']['cli_hidden'] == '4'
        assert calls['params']['cli_seed'] == 42
        assert set(calls['metrics']) == {'test_loss', 'test_accuracy'}
        assert 0.0 <= calls['metrics']['test_accuracy'] <= 1.0
        assert 'run_id: abc123' in capsys.readouterr().out

    def test_dotenv_supplies_tracking_uri(self, fast_args, tmp_path,
                                          monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('MLFLOW_TRACKING_URI=http://from-dotenv:9999\n')
        monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
        monkeypatch.setattr(train_module, 'DOTENV_PATH', env_file)

        seen = {}
        monkeypatch.setattr(train_module.mlflow, 'set_tracking_uri',
                            lambda uri: seen.__setitem__('uri', uri))
        monkeypatch.setattr(train_module.mlflow, 'set_experiment',
                            lambda name: None)
        monkeypatch.setattr(mlflow.tensorflow, 'autolog', lambda **kw: None)

        @contextlib.contextmanager
        def fake_start_run(run_name=None, **kwargs):
            yield types.SimpleNamespace(
                info=types.SimpleNamespace(run_id='x'))

        monkeypatch.setattr(train_module.mlflow, 'start_run', fake_start_run)
        monkeypatch.setattr(train_module.mlflow, 'log_params', lambda p: None)
        monkeypatch.setattr(train_module.mlflow, 'log_metrics', lambda m: None)

        train_module.main(fast_args)

        assert seen['uri'] == 'http://from-dotenv:9999'

    def test_real_env_wins_over_dotenv(self, fast_args, tmp_path,
                                       monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('MLFLOW_TRACKING_URI=http://from-dotenv:9999\n')
        monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://from-shell:1111')
        monkeypatch.setattr(train_module, 'DOTENV_PATH', env_file)

        seen = {}
        monkeypatch.setattr(train_module.mlflow, 'set_tracking_uri',
                            lambda uri: seen.__setitem__('uri', uri))
        monkeypatch.setattr(train_module.mlflow, 'set_experiment',
                            lambda name: None)
        monkeypatch.setattr(mlflow.tensorflow, 'autolog', lambda **kw: None)

        @contextlib.contextmanager
        def fake_start_run(run_name=None, **kwargs):
            yield types.SimpleNamespace(
                info=types.SimpleNamespace(run_id='x'))

        monkeypatch.setattr(train_module.mlflow, 'start_run', fake_start_run)
        monkeypatch.setattr(train_module.mlflow, 'log_params', lambda p: None)
        monkeypatch.setattr(train_module.mlflow, 'log_metrics', lambda m: None)

        train_module.main(fast_args)

        assert seen['uri'] == 'http://from-shell:1111'

    def test_custom_target_column(self, tmp_path, monkeypatch):
        frame = pd.DataFrame(
            np.random.default_rng(1).random((30, 2)), columns=['p', 'q'])
        frame['label'] = np.tile([1.0, 2.0], 15)
        path = tmp_path / 'other.csv'
        frame.to_csv(path, index=False)

        train_module.main(['--data-path', str(path), '--target', 'label',
                           '--epochs', '1', '--hidden', '4', '--verbose', '0',
                           '--no-mlflow'])
