"""Load-test scenarios for the fetal-health inference API.

Run against a locally published container with:
    locust -f locustfile.py --host http://localhost:8080
"""

from locust import HttpUser, between, task


PREDICTION_PAYLOAD = {
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
}

EXPECTED_CLASSES = {'Normal', 'Suspect', 'Pathological'}


class ApiLoadRunner(HttpUser):
    """Simulate clients checking readiness and requesting predictions."""

    wait_time = between(0.5, 2.5)

    @task(1)
    def health(self):
        self.client.get('/health', name='/health')

    @task(9)
    def predict(self):
        with self.client.post(
                '/predictions', json=PREDICTION_PAYLOAD,
                name='/predictions', catch_response=True) as response:
            if response.status_code != 200:
                response.failure(
                    f'expected HTTP 200, received {response.status_code}')
                return

            try:
                result = response.json()
            except ValueError:
                response.failure('response is not valid JSON')
                return

            if result.get('predicted_class') not in EXPECTED_CLASSES:
                response.failure('response has no recognised predicted_class')
            elif not isinstance(result.get('raw_prediction'), list):
                response.failure('response has no raw_prediction list')
            else:
                response.success()
