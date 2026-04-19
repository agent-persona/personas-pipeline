import pytest
from pathlib import Path
from persona_eval.schemas import EvalResult
from persona_eval.storage import JsonRecorder


@pytest.fixture
def tmp_recorder(tmp_path):
    return JsonRecorder(path=tmp_path / "results.jsonl")


def _make_result(dimension_id: str = "D1", passed: bool = True) -> EvalResult:
    return EvalResult(
        dimension_id=dimension_id,
        dimension_name="Test Dim",
        persona_id="p1",
        passed=passed,
        score=1.0 if passed else 0.0,
    )


def test_record_single(tmp_recorder):
    result = _make_result()
    tmp_recorder.record(result)
    loaded = tmp_recorder.load_all()
    assert len(loaded) == 1
    assert loaded[0].dimension_id == "D1"
    assert loaded[0].passed is True


def test_record_batch(tmp_recorder):
    results = [_make_result(f"D{i}", i % 2 == 0) for i in range(5)]
    tmp_recorder.record_batch(results)
    loaded = tmp_recorder.load_all()
    assert len(loaded) == 5


def test_load_empty(tmp_recorder):
    assert tmp_recorder.load_all() == []


def test_append_multiple_calls(tmp_recorder):
    tmp_recorder.record(_make_result("D1"))
    tmp_recorder.record(_make_result("D2"))
    loaded = tmp_recorder.load_all()
    assert {r.dimension_id for r in loaded} == {"D1", "D2"}
