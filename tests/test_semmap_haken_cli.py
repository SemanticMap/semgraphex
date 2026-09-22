from __future__ import annotations

import pytest

from semmap_haken import __version__
from semmap_haken.cli import main


def test_active_and_legacy_packages_are_importable() -> None:
    import semgraphex
    import semmap_haken

    assert semmap_haken.__version__ == __version__
    assert semgraphex.__name__ == "semgraphex"


def test_cli_root_and_subcommand_help(capsys) -> None:
    assert main(["--help"]) == 0
    assert "download" in capsys.readouterr().out

    assert main(["prepare", "--help"]) == 0
    assert "--config" in capsys.readouterr().out


def test_evaluate_rejects_incomplete_run() -> None:
    """Evaluate is now an active M2 evidence gate, not a deferred command."""
    with pytest.raises(ValueError, match="run is not complete"):
        main(["evaluate", "--run", "runs/missing"])
