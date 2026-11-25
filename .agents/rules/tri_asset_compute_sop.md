# Hybrid CPU-First Compute Routing SOP (Auto-Applied)

Follow the full playbook in `TRI_ASSET_COMPUTE_SETUP_AND_SOP.md` whenever running compute-heavy mining, feature sweeps, or multi-pair backtests:

1. **CPU-First Routing (Local + Headless Kaggle CPU Containers; Colab Disabled by Default):**
   - **Kaggle (`kaggle` CLI `.venv\Scripts\python.exe -m kaggle` + MCP):** Default cloud offload via 2–4 concurrent **CPU-only** script kernels (`4` Xeon vCPUs @ `2.20 GHz`, `31.35 GB` RAM each; `"enable_gpu": "false"`, `"enable_tpu": "false"`, `"is_private": "true"`, `"enable_internet": "false"`). Never burn weekly GPU/TPU quota (`30h`/`20h`) unless GPU/TPU is explicitly needed. Private Discovery dataset `devchaudharyyyy/trd-v5-discovery-m15` (`46.67 MB`, 9 pairs `.parquet`) is already hosted; preserve `.parquet` files using `-t` when updating.
   - **Kaggle Execution Guardrails (4 Proven Pitfalls):**
     1. Match `kernel-metadata.json` `"id"` to Kaggle's hyphenated slug (`trd-v5-bench-shard-1`) and parse the pushed URL from `kaggle kernels push` output before polling.
     2. Search `/kaggle/input/datasets/devchaudharyyyy/<dataset>/`, `/kaggle/input/<dataset>/`, and `Path("/kaggle/input").rglob(...)`.
     3. Only `code_file` uploads on `kernels push` — embed helpers or pull them from the attached dataset.
     4. For `< 2 min` / low-RAM Numba runs, execute locally (`workers <= 2`); offload `> 2–3 min` or high-RAM sweeps to Kaggle CPU (`~68s` cloud push/boot/download overhead buys `~3.4x` parallel compute speed + `31.35 GB` RAM per node).
   - **Google Colab (`colab-mcp`):** Keep unused unless the user explicitly asks to include Colab.
   - **Local Workstation (`D:\`, `6c/12t`, `7.28 GB` RAM):** Reserve for `<2 min` checks, final 7-gate `.npy` tick certification (`scripts/verify_strategy_candidate.py --require-stress-5of5`, `mmap=True`), and MTM portfolio simulations.

2. **Local OOM Protection (`7.28 GB` RAM Cap):**
   - Always set `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` before importing `numpy`/`pandas`/`numba`, and cap local parallel workers at `1–2` max.

3. **Non-Negotiable Quant Rules:**
   - **70/30 Split:** Screen/tune strictly on Discovery (`Weeks 1..210`); run Holdout (`Weeks 211..300`) strictly once per locked candidate.
   - **Exclude `USDJPY` permanently.**
   - **Strict Causality:** Use `tick_engine.core.causal_bars` (`closed='right', label='right'` via `resample_causal_bars`), zero `.bfill()`, and pass Gate 0 (`0` violations).
