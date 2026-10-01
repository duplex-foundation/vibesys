"""A declared capture command replaces launcher discovery in the profiler prompts."""

from __future__ import annotations

import pytest

from vibesys.prompts import PROMPTS_DIR, render_template

_AGENT_LOOP = PROMPTS_DIR / "loops" / "agent"
_CAPTURE = "python3 tools/profile.py --serving serving.json --nsys"


def _profiler_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "profile_focus": "decode kernels",
        "benchmark_command": "bash bench.sh",
        "capture_command": None,
        "modality": None,
        "domain_profiler": "",
        "runtime_notes": "",
        "profile_execution": "local",
        "objective": "Headline metric: tok_s",
        "profiler_support_name": "support",
        "profiler_mcp_name": "vibesys-profiler",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("template", ["profilers/nsys.j2", "profilers/torch.j2"])
def test_the_capture_command_is_the_only_way_to_launch_when_declared(template: str) -> None:
    prompt = render_template(
        template, template_dir=_AGENT_LOOP, **_profiler_kwargs(capture_command=_CAPTURE)
    )
    assert f"The capture command is `{_CAPTURE}`." in prompt
    assert f"    {_CAPTURE}" in prompt
    flat = " ".join(prompt.split())
    assert "Do not start the server" in flat
    assert "do not change" in flat


@pytest.mark.parametrize("template", ["profilers/nsys.j2", "profilers/torch.j2"])
def test_without_a_capture_command_the_prompt_is_unchanged(template: str) -> None:
    prompt = render_template(template, template_dir=_AGENT_LOOP, **_profiler_kwargs())
    flat = " ".join(prompt.split())
    assert "The capture command is" not in flat
    assert "Do not start the server" not in flat


def test_torch_prompt_converts_the_trace_the_capture_command_writes() -> None:
    prompt = render_template(
        "profilers/torch.j2", template_dir=_AGENT_LOOP, **_profiler_kwargs(capture_command=_CAPTURE)
    )
    assert (
        "python support/analyze_torch_profile.py convert <trace> --output /tmp/prof.json" in prompt
    )


def test_the_pre_round_decision_knows_a_profile_is_one_command() -> None:
    kwargs: dict[str, object] = {
        "objective": "Serve fast.",
        "objective_location": "OBJECTIVE.md",
        "regression_info": "",
        "exhaustion_info": "",
        "progress_location": "progress.md",
        "profiler_kind": "nsys",
        "profile_execution": "local",
        "has_history": True,
    }
    with_command = render_template(
        "orchestrator_pre_round_prompt.j2",
        template_dir=_AGENT_LOOP,
        capture_command=_CAPTURE,
        **kwargs,
    )
    assert f"The capture is one declared command, `{_CAPTURE}`" in with_command
    without = render_template(
        "orchestrator_pre_round_prompt.j2", template_dir=_AGENT_LOOP, capture_command=None, **kwargs
    )
    assert "one declared command" not in without
