"""The version is written twice, in pyproject.toml and in the package; the two must agree."""

import tomllib
from pathlib import Path

import pytest

import wvc
from wvc.cli import main

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def pyproject_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_package_version_matches_pyproject():
    assert wvc.__version__ == pyproject_version()


def test_cli_prints_the_version_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"wvc {pyproject_version()}"
