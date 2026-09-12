import os
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / 'data/reference/fetal_health.csv'
CURRENT_PATH = PROJECT_ROOT / 'data/current/fetal_health.csv'
TARGET_COLUMN = 'fetal_health'
REPORT_DIR = Path(os.environ.get('EVIDENTLY_REPORT_DIR',
                                 PROJECT_ROOT / 'reports'))


@pytest.fixture(scope='module')
def datasets():
    return pd.read_csv(REFERENCE_PATH), pd.read_csv(CURRENT_PATH)


def test_generates_data_drift_report(datasets):
    reference, current = datasets

    report = Report([DataDriftPreset()])
    result = report.run(
        reference.drop(columns=TARGET_COLUMN),
        current.drop(columns=TARGET_COLUMN))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_report = REPORT_DIR / 'data_drift_report.html'
    json_report = REPORT_DIR / 'data_drift_report.json'
    result.save_html(str(html_report))
    json_report.write_text(result.json(), encoding='utf-8')

    assert html_report.is_file() and html_report.stat().st_size > 0
    assert json_report.is_file() and json_report.stat().st_size > 0
