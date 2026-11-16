"""Conformance rule 14, this repo's own: no package imports one it is declared not to.

The template's checker ships without tests, and this rule is not the template's, so these two
hold it: it fires on an import the declared direction forbids, and stays quiet on a tree that
only names the other package in prose.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def _conformance() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conformance", REPO / "scripts/conformance.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, files: dict[str, str]) -> None:
    """A git tree declaring that `lower` must not import `upper`, with these files tracked."""
    files = {
        "pyproject.toml": '[project]\nname = "example"\n\n'
        '[tool.liulab.must-not-import]\nlower = ["upper"]\n',
        **files,
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)


@pytest.fixture(scope="module")
def conformance() -> ModuleType:
    return _conformance()


def test_rule_14_fires_on_an_import_the_declared_direction_forbids(
    conformance: ModuleType, tmp_path: Path
) -> None:
    _tree(
        tmp_path,
        {
            "src/lower/__init__.py": "",
            "src/lower/work.py": '"""Does the work."""\n\nfrom upper import helper\n',
        },
    )
    result = conformance.rule_import_direction(conformance.load(tmp_path))
    assert len(result.problems) == 1
    what, *rest = result.problems[0].splitlines()
    assert what == "src/lower/work.py:3 imports `upper` — from upper import helper"
    assert "move what this module needs out of `upper` and into `lower`" in rest[-1]


def test_rule_14_stays_quiet_on_a_tree_that_imports_the_declared_way(
    conformance: ModuleType, tmp_path: Path
) -> None:
    # `lower` names `upper` in a docstring and a comment, which is correct work: a rule looking
    # for the bare name rather than an import statement would fail this tree.
    _tree(
        tmp_path,
        {
            "src/lower/__init__.py": "",
            "src/lower/work.py": '"""Does the work. `upper` composes this one."""\n\n'
            "import os  # upper imports this too\n\nWHERE = os.sep\n",
            "src/upper/__init__.py": "",
            "src/upper/plan.py": '"""Composes `lower`."""\n\nfrom lower import work\n',
        },
    )
    result = conformance.rule_import_direction(conformance.load(tmp_path))
    assert result.problems == []
    assert result.notes == ["2 module(s) of `lower` read for `upper`"]
