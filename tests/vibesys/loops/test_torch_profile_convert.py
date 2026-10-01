"""The torch helper reads vLLM's traces: a chrome trace converts into its own schema."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_HELPER = (
    Path(__file__).resolve().parents[3]
    / "resources"
    / "profilers"
    / "torch"
    / "analyze_torch_profile.py"
)


@pytest.fixture(scope="module")
def helper():  # noqa: ANN201  # tracked: #288
    spec = importlib.util.spec_from_file_location("analyze_torch_profile", _HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TRACE = {
    "traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "flash_attn_fwd", "dur": 100.0},
        {"ph": "X", "cat": "kernel", "name": "flash_attn_fwd", "dur": 50.0},
        {"ph": "X", "cat": "kernel", "name": "gemm_fp8", "dur": 20.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "dur": 30.0},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "dur": 5.0},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 2.0},
        {"ph": "X", "cat": "python_function", "name": "forward", "dur": 999.0},
        {"ph": "i", "cat": "kernel", "name": "an instant, not a span"},
    ]
}


def test_convert_sums_events_by_name_and_category(helper, tmp_path: Path) -> None:  # noqa: ANN001
    trace = tmp_path / "rank0.pt.trace.json.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as fh:
        json.dump(_TRACE, fh)
    result = helper.convert_chrome_trace(trace)
    by_name = {e["name"]: e for e in result["events"]}
    assert by_name["flash_attn_fwd"] == {
        "name": "flash_attn_fwd",
        "category": "kernel",
        "cpu_time_us": 0.0,
        "cuda_time_us": 150.0,
        "self_cpu_time_us": 0.0,
        "self_cuda_time_us": 150.0,
        "count": 2,
    }
    assert by_name["aten::linear"]["category"] == "operator"
    assert by_name["aten::linear"]["self_cpu_time_us"] == 30.0
    assert by_name["Memcpy HtoD"]["category"] == "memory"
    assert by_name["cudaLaunchKernel"]["category"] == "cpu"
    assert "forward" not in by_name, "python frames are not events"
    assert result["total_cuda_time_us"] == 175.0
    assert result["total_cpu_time_us"] == 32.0
    assert result["source"] == str(trace)


def test_the_converted_file_feeds_the_analysis_tools(helper, tmp_path: Path, capsys) -> None:  # noqa: ANN001
    trace = tmp_path / "rank0.pt.trace.json"
    trace.write_text(json.dumps(_TRACE))
    out = tmp_path / "prof.json"
    sys.argv = ["analyze_torch_profile.py", "convert", str(trace), "--output", str(out)]
    helper.main()
    sys.argv = ["analyze_torch_profile.py", "kernels", str(out), "--top", "1"]
    helper.main()
    printed = capsys.readouterr().out
    assert "flash_attn_fwd" in printed
    assert "gemm_fp8" not in printed


def test_newest_trace_is_the_last_one_written(helper, tmp_path: Path) -> None:  # noqa: ANN001
    old = tmp_path / "a.pt.trace.json.gz"
    old.write_bytes(b"")
    new = tmp_path / "sub" / "b.pt.trace.json.gz"
    new.parent.mkdir()
    new.write_bytes(b"")
    os.utime(old, (1, 1))
    assert helper.newest_trace(tmp_path) == new
    with pytest.raises(FileNotFoundError):
        helper.newest_trace(tmp_path / "sub" / "empty-dir-that-does-not-exist")
