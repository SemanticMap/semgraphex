# ConceptNet 1M · parallel CPU + CUDA-first port-factorization pilot

Open [the Colab notebook](../notebooks/08_conceptnet_1m_port_factorization_cuda_colab.ipynb) on the branch `feature/conceptnet-1m-cuda-port-pilot`.

This is a **sampled prototype experiment**, not a 1-million-node lossless
graph codec. It compares the 100k control and a new 1M connected ConceptNet
5.7 subgraph at the same random-center budgets: 500, 1,000, 3,000, 5,000.
Original exact types include typed external ports. NetworkX VF2 verifies
internal typed-graph isomorphism. Each variant retains port signatures and
the exact shape-to-prototype node permutation, and every stored prototype
is round-tripped before reporting byte savings.

## Parallelism and CUDA contract

| Stage | Execution |
| --- | --- |
| Official ConceptNet extraction (three passes) | Bounded CPU process pool, `--workers` |
| Ego extraction from sparse graph | CPU thread pool, `EgoExtractor.extract_centers(workers=...)` |
| Typed WL fingerprints for dictionary identity | Bounded spawn CPU process pool; deterministic ordered map |
| Exact dictionary VF2 and internal shape factorization | CPU, sequential (not advertised as CUDA) |
| Typed WL features for kNN | CPU feature hashing, optional CPU workers |
| Exact cosine kNN for Wishart k=4/5 | **CUDA when available**; otherwise explicitly reported CPU |

`--device auto` is the default. It selects `cuda` if PyTorch reports a
usable GPU; `--device cuda` requires CUDA and fails without one. The pilot
verifies that selected CUDA truly produced the `torch_cuda` kNN backend and
fails if the backend silently changes. No GPU claim is made for VF2, graph
construction or process-pool fingerprint calculation. CPU subprocesses finish
before GPU kNN starts to avoid CUDA/fork interactions.

See `report.json.parallel` for the actual process backend and observed worker
PIDs and `report.json.wishart` for selected device and actual kNN backend.

## Running without Colab

After preparing exactly 1,000,000 selected English concept nodes in a TSV:

```bash
python -m semmap_haken.million_port_pilot \
  --dataset /path/to/conceptnet_en_1m.tsv \
  --target-nodes 1000000 --output /path/to/new-output \
  --sample-centers 5000 --budgets 500 1000 3000 5000 \
  --workers 4 --device auto --gpu-batch-size 128
```

Results use a fresh Drive directory
`analysis/conceptnet-1m-port-cuda-pilot-v2/{100k,1m}/`. The notebook never
overwrites the original 100k runs. It stages large I/O on the Colab scratch
disk, saves validated results and writes `COMPLETED` last.

**Limitations:** The existing graph builder materializes selected assertions
in Python. A million nodes may need more than the 36-GiB preflight threshold;
use High-RAM and reduce worker count if memory is tight. Extracting 1M from the
official dump requires three passes and currently has no mid-pass resume.
`json_saved_percent` and gzip byte counts cover stored prototypes, not the
quotient/residual graph or actual external endpoint IDs. Cluster counts at
k=4/5 are diagnostics, not evidence of a full graph compression gain.
