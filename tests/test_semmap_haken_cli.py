from __future__ import annotations

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


def test_unimplemented_command_has_clear_error(capsys) -> None:
    assert main(["run", "--config", "configs/conceptnet_en_smoke.yaml"]) == 2
    captured = capsys.readouterr()
    assert "not yet implemented" in captured.err
