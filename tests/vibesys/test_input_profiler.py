"""The optional ``[profiler] capture_command`` manifest key.

A task that knows how to serve and load its own system under a profiler
declares the command once; the profiler role runs it as is instead of
discovering a launcher. It follows the evaluator commands' argv rule (a bare
name or a file inside the project), renders back out of the manifest, and is
absent, not empty, when undeclared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibesys.input_manifest import load_input_bundle, render_input_manifest

if TYPE_CHECKING:
    from pathlib import Path

_BASE = """
version = 1

[agent]
domain = "llm-serving"

[accuracy]
command = ["bash", "check.sh"]

[benchmark]
command = ["bash", "bench.sh"]
"""


def _bundle(root: Path, extra: str = "") -> Path:
    (root / "OBJECTIVE.md").write_text("Serve fast.\n")
    (root / "vibesys.input.toml").write_text(_BASE.lstrip() + extra)
    return root


def test_the_capture_command_is_loaded_resolved_and_rendered(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "profile.py").write_text("")
    bundle = load_input_bundle(
        _bundle(
            tmp_path,
            '\n[profiler]\ncapture_command = ["python3", "tools/profile.py", "--nsys", "a b"]\n',
        )
    )
    assert bundle.resolved_capture_command == ("python3", "tools/profile.py", "--nsys", "a b")
    assert bundle.capture_command_display == "python3 tools/profile.py --nsys 'a b'"
    rendered = render_input_manifest(bundle.manifest)
    assert (
        '[profiler]\ncapture_command = ["python3", "tools/profile.py", "--nsys", "a b"]' in rendered
    )


def test_a_bundle_without_the_key_has_no_capture_command(tmp_path: Path) -> None:
    bundle = load_input_bundle(_bundle(tmp_path))
    assert bundle.resolved_capture_command is None
    assert bundle.capture_command_display is None
    assert "[profiler]" not in render_input_manifest(bundle.manifest)


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ('["/usr/bin/python3", "x"]', "must be relative to the project"),
        ('["../elsewhere/profile.py"]', "escapes the project"),
        ('["tools/missing.py"]', "does not exist"),
        ("[]", "at least one argv element"),
        ('["python3", ""]', "non-empty strings"),
    ],
)
def test_the_capture_command_follows_the_evaluator_commands_argv_rule(
    tmp_path: Path, command: str, message: str
) -> None:
    (tmp_path / "tools").mkdir()
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_input_bundle(_bundle(tmp_path, f"\n[profiler]\ncapture_command = {command}\n"))
