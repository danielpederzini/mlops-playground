import asyncio

import numpy as np

from app import main as app_module


class StubModel:
    def decision_function(self, input_array):
        assert input_array.shape == (1, 21)
        return np.array([[0.01, 0.09, 0.90]])

    def predict(self, input_array):
        return np.array([2])


def test_predict_returns_the_complete_prediction_result(monkeypatch):
    monkeypatch.setattr(app_module, 'model', StubModel(), raising=False)
    request = app_module.FetalHealthInferenceRequest.model_validate({
        'baseline value': 134.0,
        'accelerations': 0.001,
        'fetal_movement': 0.0,
        'uterine_contractions': 0.01,
        'light_decelerations': 0.009,
        'severe_decelerations': 0.0,
        'prolongued_decelerations': 0.002,
        'abnormal_short_term_variability': 26.0,
        'mean_value_of_short_term_variability': 5.9,
        'percentage_of_time_with_abnormal_long_term_variability': 0.0,
        'mean_value_of_long_term_variability': 0.0,
        'histogram_width': 150.0,
        'histogram_min': 50.0,
        'histogram_max': 200.0,
        'histogram_number_of_peaks': 5.0,
        'histogram_number_of_zeroes': 3.0,
        'histogram_mode': 76.0,
        'histogram_mean': 107.0,
        'histogram_median': 107.0,
        'histogram_variance': 170.0,
        'histogram_tendency': 0.0,
    })

    result = asyncio.run(app_module.predict(request))

    assert result == {
        'predicted_class': 'Pathological',
        'raw_prediction': [[0.01, 0.09, 0.9]],
    }
