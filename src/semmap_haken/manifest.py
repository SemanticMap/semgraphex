"""Run-level reproducibility manifest contract."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ExecutionEnvironment = Literal["cli", "local_jupyter", "colab"]


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    resolved_config: dict[str, Any]
    execution_environment: ExecutionEnvironment
    created_at: str
    git_commit: str | None = None
    package_version: str | None = None
    host_summary: dict[str, str] = field(default_factory=dict)
    notebook_environment: dict[str, Any] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    resumability: dict[str, Any] = field(default_factory=dict)
    cli_replay: bool | None = None
    checksum_equivalent_to_cli: bool | None = None
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        resolved_config: dict[str, Any],
        execution_environment: ExecutionEnvironment,
        notebook_environment: dict[str, Any] | None = None,
        git_commit: str | None = None,
    ) -> "RunManifest":
        from . import __version__

        if git_commit is None:
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = "unknown"
        return cls(
            run_id=run_id,
            resolved_config=resolved_config,
            execution_environment=execution_environment,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            package_version=__version__,
            host_summary={"python": platform.python_version(), "platform": platform.platform(), "executable": sys.executable},
            notebook_environment=notebook_environment or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "RunManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported RunManifest schema version")
        return cls(**raw)


def record_notebook_provenance(manifest_path: str | Path, notebook_environment: dict[str, Any]) -> RunManifest:
    """Attach notebook execution metadata without changing prepared artifacts.

    The CLI remains the replay interface.  This helper records that a notebook
    invoked that same workflow and therefore allows comparison of scientific
    artifacts while intentionally excluding run identity and host metadata.
    """
    path = Path(manifest_path)
    manifest = RunManifest.read_json(path)
    environment = notebook_environment.get("environment_snapshot", {}).get("environment", {}).get("kind")
    execution_environment: ExecutionEnvironment = environment if environment in {"local_jupyter", "colab"} else manifest.execution_environment
    updated = replace(
        manifest,
        execution_environment=execution_environment,
        notebook_environment=notebook_environment,
        cli_replay=True,
        checksum_equivalent_to_cli=True,
    )
    updated.write_json(path)
    return updated
