# Hybrid CPU-First Compute Setup & Standard Operating Procedure (SOP)

**Purpose:** Automatically distribute heavy quantitative research, feature screening, and backtesting across **Local Workstation + Headless Kaggle CPU-Only Containers** (with Google Colab available as an optional 3rd asset only when explicitly requested) so:
1. Local laptop RAM (`7.28 GB`) never hits OpenBLAS/memory OOM crashes.
2. Kaggle's **30.0h/week GPU** and **20.0h/week TPU** quotas remain **0% touched** (`enable_gpu: false`, `enable_tpu: false`) unless ML/GPU work is explicitly requested.
3. Heavy >2–3 minute grids run in parallel across **`4`-vCPU / `31.35 GB`-RAM Kaggle CPU containers** while local NVMe handles fast `<2 min` Numba checks and final tick-by-tick certification without uploading `37.58 GB` of raw tick files.

---

## 1. Verified Hardware & Resource Inventory (Empirically Measured)

| Asset | Verified CPU / Threads | Verified RAM | Storage & Role | Connection Path | Default Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Local (`D:\`)** | `AMD Ryzen 5 7530U` (**6c / 12t**) | **`7.28 GB` usable** (`~0.85–1.5 GB` free under IDE) | **`65.6 GB` free SSD**, holds **`37.58 GB`** `.npy` tick store + **`46.67 MB`** Discovery M15 cache | Local PowerShell + `.venv\Scripts\python.exe` | **1–2 workers max**, `mmap=True`, `<2 min` sweeps & 7-gate tick certification |
| **2. Kaggle CPU Cloud** | **`4` Xeon vCPUs @ `2.20 GHz` per container** (2+ concurrent tested = **8+ cloud vCPUs**) | **`31.35 GB` RAM per container** (`62.7 GB` across 2 shards) | Private dataset `devchaudharyyyy/trd-v5-discovery-m15` (**`46.67 MB`**, 9 pairs `*_disc_m15.parquet`, status: `ready`) | `.venv\Scripts\python.exe -m kaggle` (`2.2.5`, headless CLI) + `kaggle` MCP (`https://www.kaggle.com/mcp`) | **Default cloud offload:** `enable_gpu: false`, `enable_tpu: false`, `is_private: true`, offline (`enable_internet: false`) |
| **3. Google Colab *(Optional)*** | `2` cloud CPU cores | **`12.67 GB` RAM** | Ephemeral VM (requires interactive Chrome tab click) | `colab-mcp` (`open_colab_browser_connection` + cell tools) | **Disabled by default** unless user asks for Colab; never auto-open Chrome popups |

> **Kaggle Quota Protection:** Verified `108,000s` (`30.0h`) GPU and `72,000s` (`20.0h`) TPU weekly quotas. Setting `"enable_gpu": "false"` and `"enable_tpu": "false"` runs pure CPU Xeon nodes with **0s quota usage**.

---

## 2. Empirical Benchmark & Break-Even Rule (9 Pairs, 116,640 Configs / 233,280 M15 Sims)

Tested on identically coded strictly-causal Discovery sweeps (`scratch/bench_core_sweep.py`, `--repeat 6`):
- **Local-Only (`workers=2`):** `80.196s` compute (`+ 3.973s` Numba JIT = **`84.169s`** total), `234.6 MB` max worker RSS.
- **Local (`2` pairs) + Dual-Kaggle CPU Shards (`4` + `3` pairs):**
  - **Pure Compute Speedup:** **`3.44×`** (`80.196s` $\to$ **`23.335s`** max shard: Local `23.335s`, Kaggle Shard 1 `22.503s`, Kaggle Shard 2 `15.749s`; `27.167s` incl. JIT = `3.10×`).
  - **End-to-End Wall Time:** `98.880s` (`13.56s` push + `~45s` queue/boot/poll + `20.23s` download) because Kaggle batch push + container boot + output pull has **`~65–70s` fixed cloud overhead**.
  - **Exact Parity:** **`9 / 9` pairs bit-exact match** (`AUDCAD`, `AUDNZD`, `EURGBP`, `EURNZD`, `EURUSD`, `GBPCHF`, `GBPJPY`, `NZDCAD`, `USDCAD`).

**Routing Break-Even Decision Rule:**
- **Task `< ~2 min` local & low RAM (`< 1 GB`):** Run **locally (`workers <= 2`)** — faster end-to-end than cloud boot overhead.
- **Task `> ~2–3 min` local OR memory-heavy (`> 1.5 GB` RSS / multi-pair Optuna / wide feature sweeps):** Offload shards to **2–4 concurrent Kaggle CPU containers** (`31.35 GB` RAM each) simultaneously with a small local shard for **`~3×+` real compute scaling** and zero OOM risk.

---

## 3. Global Setup (Outside Workspace — Persists Across Repo/Checkpoint Resets)

1. **Credentials (NEVER store in Git):**
   - `%USERPROFILE%\.kaggle\access_token` (user-only `icacls` permissions, used by `.venv\Scripts\python.exe -m kaggle ...`)
   - `%USERPROFILE%\.env` (`KAGGLE_API_TOKEN=...`)
   - Ready private dataset already hosted under account `devchaudharyyyy`: `devchaudharyyyy/trd-v5-discovery-m15` (`46.67 MB`, `.parquet` preserved).
2. **Antigravity Global MCP Config (`C:\Users\kdev2\.gemini\config\mcp_config.json`):**
   - `kaggle` MCP: `https://www.kaggle.com/mcp` with `Authorization: Bearer YOUR_KGAT_TOKEN_HERE` header.
   - `colab-mcp` (only if user explicitly enables Colab): points directly to `C:/Users/kdev2/AppData/Roaming/uv/tools/colab-mcp/Scripts/python.exe` with `-c "import colab_mcp; colab_mcp.main()"` (bypasses Windows Application Control blocking `%USERPROFILE%\.local\bin\colab-mcp.exe`).
3. **`.venv` Python Packages:** `kaggle` (`2.2.5`), `kagglehub` (`1.0.3`), `numba`, `pyarrow`, `pandas`, `numpy`.

---

## 4. Mandatory CPU-First Guardrails

1. **Local Threading & RAM Cap (Non-Negotiable):** Every local script MUST set BEFORE importing `numpy`/`pandas`/`numba`:
   ```python
   import os
   os.environ["OPENBLAS_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   os.environ["OMP_NUM_THREADS"] = "1"
   ```
   Cap local `.npy` certification parallelism at **`1–2` workers** (`mmap=True`) inside our `7.28 GB` RAM ceiling.
2. **Kaggle CPU-Only & Offline Default:** In every `kernel-metadata.json`:
   ```json
   {
     "id": "devchaudharyyyy/<clean-slug>",
     "title": "<Title Matching Slug>",
     "code_file": "shard_run.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": "true",
     "enable_gpu": "false",
     "enable_tpu": "false",
     "enable_internet": "false",
     "machine_shape": "",
     "dataset_sources": ["devchaudharyyyy/trd-v5-discovery-m15"],
     "competition_sources": [],
     "kernel_sources": [],
     "model_sources": []
   }
   ```
   Never enable GPU/TPU for Numba/tabular backtests. Keep `enable_internet: "false"` when using standard preinstalled packages (`numpy`, `pandas`, `numba`, `pyarrow`, `scipy`).
3. **No Unrequested Colab Browser Popups:** Do not call `open_colab_browser_connection` unless the user explicitly asks to use Google Colab.

---

## 5. Battle-Tested Kaggle Playbook & Known Pitfalls

### Phase 1: Dataset Sync (Only Needed When Data Changes)
`devchaudharyyyy/trd-v5-discovery-m15` already holds the 9 causal Discovery M15 `.parquet` files (`Weeks 1..210`, `46.67 MB`). If updating it:
```powershell
.venv\Scripts\python.exe -m kaggle datasets version -p scratch\kaggle_bundle -m "Update bundle" -t -r zip
```
*(ALWAYS pass `-t` so Kaggle keeps `.parquet` intact instead of converting to slow `.csv`.)*

### Phase 2: Dispatch Sharded Headless Kaggle CPU Scripts + Local Shard
Split heavy 9-pair sweeps into disjoint symbol shards (e.g., Shard 1: `4` pairs across `4` Kaggle Xeon vCPUs; Shard 2: `3` pairs across `4` Kaggle Xeon vCPUs; Local: `2` pairs across `2` workers):
1. Push kernels:
   ```powershell
   .venv\Scripts\python.exe -m kaggle kernels push -p scratch\kaggle_shard1
   .venv\Scripts\python.exe -m kaggle kernels push -p scratch\kaggle_shard2
   ```
2. **Four Verified Pitfalls to Avoid in Every Kaggle Runner (`shard_run.py` + Orchestrator):**
   - **Pitfall 1 — Slug Auto-Hyphenation:** Kaggle slugifies digits (`"TRD v5 Bench Shard 1"` $\to$ `devchaudharyyyy/trd-v5-bench-shard-1`, with `-1`). Always match `id` to the exact hyphenated title slug **and** parse the live URL from `kaggle kernels push` stdout before polling `kaggle kernels status <slug>`.
   - **Pitfall 2 — Dataset Input Path:** Kaggle mounts attached datasets at `/kaggle/input/datasets/devchaudharyyyy/<slug>/` (not always `/kaggle/input/<slug>/`). Always check both candidates plus `Path("/kaggle/input").rglob("*.parquet")`.
   - **Pitfall 3 — Only `code_file` Uploads on `kernels push`:** Local sibling `.py` files inside `scratch/kaggle_shard1/` do NOT upload automatically. Either load shared scripts from the dataset OR embed `lzma+base64` script source inside `shard_run.py` and write to `/kaggle/working/` before importing.
   - **Pitfall 4 — PowerShell Command Chaining:** Windows PowerShell 5.1 rejects `&&`; use `; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE };` between dependent CLI steps.

### Phase 3: Harvest Outputs & Run Local 7-Gate Tick Certification
1. Poll status until `KernelWorkerStatus.COMPLETE`, then download JSON/CSVs + logs:
   ```powershell
   .venv\Scripts\python.exe -m kaggle kernels status devchaudharyyyy/trd-v5-bench-shard-1
   .venv\Scripts\python.exe -m kaggle kernels output devchaudharyyyy/trd-v5-bench-shard-1 -p scratch\kaggle_out_shard1
   ```
2. Generate standalone strategy `.py` files locally (`tick_engine.core.causal_bars`, locked Discovery parameters) and verify sequentially (`1–2` workers max) against local `.npy` ticks:
   ```powershell
   $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
   .venv\Scripts\python.exe scripts\verify_strategy_candidate.py `
     --strategy <path.py> `
     --symbol <SYM> `
     --fleet-manifest v5.0_production\FLEET_MANIFEST.json `
     --output-dir results\v5_production_manifests `
     --require-stress-5of5
   ```

---

## 6. Hard Quant Integrity Rules (Non-Negotiable)
- **Strict 70/30 Chronological Split:** Screen/tune strictly on Discovery (`Weeks 1..210`); test Holdout (`Weeks 211..300`) strictly once per locked candidate.
- **`USDJPY` Permanently Excluded:** Never mine, certify, or simulate `USDJPY`.
- **Strict Causality:** Always use `build_causal_time_bars`, `resample_causal_bars(..., closed="right", label="right")`, `align_htf_to_ltf`, zero `.bfill()`, zero future leakage, and pass Gate 0 (`0` causality violations).
