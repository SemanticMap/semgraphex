"""nSBM fit checkpoints must reject same-size tampering."""
from pathlib import Path
import numpy as np

from semmap_haken.hypothesis import runner
from test_nsbm_huffman_experiment import sample


def test_fit_checkpoint_checksum_after_same_size_mutation(tmp_path):
    graph = sample()
    path = tmp_path / "graph"
    runner._save_graph(graph, path)
    task = tmp_path / "nested_restart_000.json"

    def fake_fit(g, *, kind, seed, threads):
        return {
            "kind": kind, "seed": seed, "mdl_bits": 5.,
            "partitions": [[i // 4 for i in range(g.vertex_count)]],
            "blocks_by_level": [3],
        }

    runner._fit_task(path, "nested-DC-SBM", 42, 1, task,
                     injected_fit=fake_fit)
    checkpoint = task.with_suffix(".npz")
    assert runner._valid_task(task, checkpoint, task_key=task.stem)
    data = bytearray(checkpoint.read_bytes())
    data[len(data)//2] ^= 1
    checkpoint.write_bytes(data)
    assert not runner._valid_task(task, checkpoint, task_key=task.stem)
