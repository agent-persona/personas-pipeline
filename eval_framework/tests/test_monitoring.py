"""Tests for production monitoring."""

import pytest
from unittest.mock import MagicMock, patch
from persona_eval.schemas import EvalResult


def test_monitor_importable():
    from persona_eval.monitoring import ProductionMonitor
    assert ProductionMonitor is not None


def test_stable_metrics_no_alert():
    from persona_eval.monitoring import ProductionMonitor
    # Build stable history
    history = [
        EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                   passed=True, score=round(0.9 + i * 0.01, 2))
        for i in range(5)
    ]
    current = EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                         passed=True, score=0.91)

    alerter = MagicMock()
    monitor = ProductionMonitor(alerter=alerter)
    result = monitor.check_drift("D1", history, current)
    assert result["status"] == "stable"
    alerter.alert_regression.assert_not_called()


def test_drift_detected_and_alerted():
    from persona_eval.monitoring import ProductionMonitor
    # Stable history around 0.9
    history = [
        EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                   passed=True, score=0.9)
        for _ in range(5)
    ]
    # Current much lower
    current = EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                         passed=False, score=0.3)

    alerter = MagicMock()
    monitor = ProductionMonitor(alerter=alerter)
    result = monitor.check_drift("D1", history, current)
    assert result["status"] == "drift_detected"
    alerter.alert_regression.assert_called_once()


def test_insufficient_history():
    from persona_eval.monitoring import ProductionMonitor
    history = [
        EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                   passed=True, score=0.9)
    ]
    current = EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                         passed=True, score=0.5)

    monitor = ProductionMonitor()
    result = monitor.check_drift("D1", history, current)
    assert result["status"] == "insufficient_data"


def test_improving_no_alert():
    from persona_eval.monitoring import ProductionMonitor
    history = [
        EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                   passed=True, score=0.7)
        for _ in range(5)
    ]
    current = EvalResult(dimension_id="D1", dimension_name="Test", persona_id="p1",
                         passed=True, score=0.95)

    alerter = MagicMock()
    monitor = ProductionMonitor(alerter=alerter)
    result = monitor.check_drift("D1", history, current)
    assert result["status"] == "stable"
    alerter.alert_regression.assert_not_called()
