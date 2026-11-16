from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from wevisone_vl import cli


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


class _FakeEngine:
    error: Exception | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, image, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(response="ok")


def _run(monkeypatch, tmp_path: Path, image: Path | str, *extra: str) -> int:
    monkeypatch.setattr(
        "sys.argv",
        ["wevisone-vl", "--image", str(image), "--output-dir", str(tmp_path / "out"), *extra],
    )
    return cli.main()


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    _FakeEngine.error = None
    monkeypatch.setattr(cli, "WeVisOneVL", _FakeEngine)
    # cli.main() saves through the module-level _save_result wrapper (the
    # supervision render stack is imported lazily inside it), so patching
    # the wrapper keeps the suite runnable on hosts without supervision.
    monkeypatch.setattr(
        cli,
        "_save_result",
        lambda output_dir, image, result: {"metadata": Path(output_dir) / "metadata.json"},
    )


def test_success_returns_zero(monkeypatch, tmp_path, image_file, capsys) -> None:
    assert _run(monkeypatch, tmp_path, image_file) == 0
    assert "ok" in capsys.readouterr().out


def test_missing_image_is_a_usage_error(monkeypatch, tmp_path, caplog) -> None:
    with caplog.at_level(logging.ERROR):
        code = _run(monkeypatch, tmp_path, tmp_path / "missing.png")
    assert code == cli.EXIT_USAGE
    assert "not an existing file" in caplog.text


@pytest.mark.parametrize(
    "flag", [("--max-new-tokens", "0"), ("--decoder-batch-size", "-1"), ("--temperature", "-0.1")]
)
def test_invalid_numeric_arguments_are_rejected(monkeypatch, tmp_path, image_file, flag) -> None:
    assert _run(monkeypatch, tmp_path, image_file, *flag) == cli.EXIT_USAGE


def test_corrupt_image_is_reported_without_a_traceback(monkeypatch, tmp_path, caplog) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with caplog.at_level(logging.ERROR):
        assert _run(monkeypatch, tmp_path, bad) == cli.EXIT_FAILURE
    assert "Cannot read image" in caplog.text


def test_model_init_failure_is_logged(monkeypatch, tmp_path, image_file, caplog) -> None:
    def broken(*args, **kwargs):
        raise OSError("weights not found")

    monkeypatch.setattr(cli, "WeVisOneVL", broken)
    with caplog.at_level(logging.ERROR):
        assert _run(monkeypatch, tmp_path, image_file) == cli.EXIT_FAILURE
    assert "Failed to initialize model" in caplog.text
    assert "weights not found" in caplog.text


def test_inference_failure_is_logged(monkeypatch, tmp_path, image_file, caplog) -> None:
    _FakeEngine.error = RuntimeError("CUDA out of memory")
    with caplog.at_level(logging.ERROR):
        assert _run(monkeypatch, tmp_path, image_file) == cli.EXIT_FAILURE
    assert "Inference failed" in caplog.text


def test_save_failure_still_prints_the_response(
    monkeypatch, tmp_path, image_file, caplog, capsys
) -> None:
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "_save_result", broken)
    with caplog.at_level(logging.ERROR):
        assert _run(monkeypatch, tmp_path, image_file) == cli.EXIT_FAILURE
    assert "Cannot write results" in caplog.text
    assert "ok" in capsys.readouterr().out


def test_keyboard_interrupt_during_inference(monkeypatch, tmp_path, image_file) -> None:
    _FakeEngine.error = KeyboardInterrupt()
    assert _run(monkeypatch, tmp_path, image_file) == cli.EXIT_INTERRUPTED
