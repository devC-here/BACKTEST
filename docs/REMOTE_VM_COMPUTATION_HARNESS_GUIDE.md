# 🚀 Azure VM Remote Computation Harness Guide for BACKTEST Engine

> **Operational Blueprint**: How to offload, parallelize, and scale heavy computational workloads (Alpha Discovery Battery, 5-Year High-Speed Tick Audits, Optuna Hyperparameter Optimization, and Monte Carlo Sweeps) from your local machine to the Azure Cloud VM (`azure-vm`).

---

## 🏛️ System Architecture: Local Studio vs. Cloud Compute Engine

```
+-------------------------------------------------------------+          +-------------------------------------------------------------+
|               LOCAL WORKSTATION (DEVELOPMENT)               |          |                 AZURE CLOUD VM (EXECUTION)                  |
|          Path: D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST   |          |                     Path: C:\BACKTEST                       |
+-------------------------------------------------------------+          +-------------------------------------------------------------+
|  • Strategy Prototyping & Alpha Hypothesis Formulation      |          |  • 24/7 Uninterrupted Background Execution Loop             |
|  • Code Editing, Vector Formulation & Unit Smoke Tests      |   SSH    |  • Numba @njit Multi-Core Parallel Batch Simulation         |
|  • Interactive Results Analysis, Charting & Reporting       | -------> |  • 5-Year 10-Block In-Sample / Out-of-Sample Holdout Audits |
|  • Machine remains cool, silent, and responsive for work    | (Port 22)|  • Massive Alpha Discovery Sweeps across 10 Forex Pairs     |
+-------------------------------------------------------------+          +-------------------------------------------------------------+
                               ^                                                                        |
                               |                       SCP / SFTP Artifact Pull                         |
                               +------------------------------------------------------------------------+
                                         (Pulls results/, logs/, metrics.csv, and charts back)
```

### Why Harness the VM for Computation?
1. **Uninterrupted 24/7 Processing**: Massive discovery runs across 10 currency pairs or 5-year tick history can take hours. Running them on the VM allows your local laptop to be disconnected, moved, or put to sleep without aborting the simulation.
2. **On-Demand Elastic Scaling (The Cloud Multiplier)**: When you need to run an intensive grid search or 100-million-tick discovery battery, Azure allows you to vertically scale the VM from its standard 2 vCPUs up to **8, 16, 32, or 64 vCPUs** (`Standard_D16s_v5` or `Standard_F16s_v2`) for $0.50–$1.00/hour, complete your backtest in minutes instead of overnight, and then scale back down to baseline.
3. **Dedicated NVMe I/O & Memory-Mapped Ticks**: Zero-copy `.npy` memory maps execute at up to **35 Million ticks/second** without competing with local desktop applications for RAM or disk throughput.
4. **Coexistence with Live Trading (`H1-ONLINE`)**: Computational jobs run isolated in `C:\BACKTEST` with `BelowNormal` process priority, ensuring the live paper trading fleet (`H1-ONLINE`, Port 5001) never drops a single tick or candle evaluation.

---

## ⚡ STEP 1: One-Time VM Environment Initialization

Before running your first remote job, execute the one-time directory creation and dependency installation on the Azure VM.

### Option A: 1-Click Automated Setup (from Local PowerShell)
Run this single command from your local machine:
```powershell
ssh azure-vm "powershell -Command \"New-Item -ItemType Directory -Force -Path C:\BACKTEST, C:\BACKTEST\data, C:\BACKTEST\results, C:\BACKTEST\scripts, C:\DATA-TICK ; pip install numba optuna duckdb polars ta\""
```

### Option B: Step-by-Step Manual Commands
1. Connect to the Azure VM via SSH:
   ```bash
   ssh azure-vm
   ```
2. Create the workspace directories:
   ```cmd
   mkdir C:\BACKTEST
   mkdir C:\BACKTEST\data
   mkdir C:\BACKTEST\results
   mkdir C:\BACKTEST\scripts
   mkdir C:\DATA-TICK
   ```
3. Install required computational packages into the VM's Python environment:
   ```cmd
   pip install numba optuna duckdb polars ta
   ```
4. Verify Numba fastmath compilation and Python setup:
   ```cmd
   python -c "import numba, numpy, pyarrow, pandas, optuna; print('VM Computational Environment Ready! Numba:', numba.__version__)"
   ```

---

## 🔄 STEP 2: Synchronizing Code and Tick Datasets

### A. Syncing the `BACKTEST` Codebase (Excluding Heavy Raw Data)
To copy your strategies, engine code, and discovery suites to the VM:
```powershell
# From local D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST:
scp -r tick_engine data_engineering_suite *.py requirements.txt azure-vm:C:/BACKTEST/
```

### B. Syncing Tick Datasets (`data/` or `DATA-TICK`)
Since tick files (`.csv` or `.npy`) are large (several gigabytes), you have two flexible methods:

#### Method 1: Sync a Single Symbol (Targeted & Fast)
If you are optimizing or discovering alphas for a specific pair (e.g. `EURUSD` or `USDJPY`), sync only that symbol's folder:
```powershell
# Creates symbol folder on VM and uploads its weekly files
ssh azure-vm "mkdir C:\BACKTEST\data\EURUSD"
scp -r "D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST\data\EURUSD\*" azure-vm:C:/BACKTEST/data/EURUSD/
```

#### Method 2: Sync Processed Memory-Mapped `.npy` Arrays (Zero-Copy)
If you already generated `.npy` files locally in `tick_engine/data/processed/`, uploading `.npy` files is significantly faster than uploading raw CSVs because binary files transfer cleanly and avoid remote conversion time:
```powershell
ssh azure-vm "mkdir C:\BACKTEST\tick_engine\data\processed"
scp -r "D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST\tick_engine\data\processed\*" azure-vm:C:/BACKTEST/tick_engine/data/processed/
```

---

## 🛠️ STEP 3: The 4 Remote Execution Workflows

### WORKFLOW 1: 1-Click Automated Dispatcher (Recommended)
We provide a dedicated orchestrator script [`scripts/remote_compute_dispatcher.py`](../BACKTEST/scripts/remote_compute_dispatcher.py) and a 1-click batch launcher [`dispatch_to_vm.bat`](../BACKTEST/dispatch_to_vm.bat).

#### How to Run:
```powershell
# Run Master Symbol Alpha Discovery on EURUSD on the VM:
python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURUSD"

# Run 5-Year Fleet Audit on the VM:
python scripts/remote_compute_dispatcher.py --script run_5y_fleet_audit.py

# Custom Hurdle Rate & High-Core Concurrency:
python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol GBPJPY --hurdle-return 18.0 --hurdle-pf 1.30 --max-workers 8"
```

#### What the Dispatcher Does Automatically:
1. Syncs the latest local strategy code, scripts, and configs to `azure-vm:C:\BACKTEST\`.
2. Initiates execution on the VM with output redirection to `C:\BACKTEST\job.log`.
3. Streams the real-time tick-processing progress bars and milestone tables back to your local terminal.
4. Automatically downloads all newly generated artifacts (`results/*.csv`, `results/*.png`, `results/*.json`) back to your local `results/` folder upon completion.

---

### WORKFLOW 2: Unattended Background Execution (Fire-and-Forget / Overnight Runs)
When you want to kick off a massive multi-hour job and turn off your laptop:

1. **Launch the background job on the VM**:
   ```powershell
   ssh azure-vm "cd C:\BACKTEST && powershell -Command \"Start-Process python -ArgumentList 'run_symbol_alpha_discovery.py --symbol EURUSD' -RedirectStandardOutput 'job_eurusd.log' -RedirectStandardError 'job_eurusd_err.log' -NoNewWindow\""
   ```
2. **Close your laptop or disconnect**. The job continues executing inside Windows on the cloud VM.
3. **Inspect progress anytime from any terminal**:
   ```powershell
   # Check if Python is actively computing:
   ssh azure-vm "Get-Process python | Select-Object Id, CPU, WorkingSet64"

   # Tail the last 30 lines of the live log:
   ssh azure-vm "powershell -Command \"Get-Content C:\BACKTEST\job_eurusd.log -Tail 30\""
   ```
4. **Pull results once completed**:
   ```powershell
   scp -r azure-vm:C:/BACKTEST/results/* "D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST\results\"
   ```

---

### WORKFLOW 3: Interactive Cloud Studio (VS Code Remote-SSH)
You can attach your local Visual Studio Code directly to `C:\BACKTEST` on the VM for an interactive, IDE-native experience:

1. Install the **Remote - SSH** extension in VS Code.
2. Press `Ctrl + Shift + P` -> Select **Remote-SSH: Connect to Host...** -> Choose `azure-vm`.
3. Select **File -> Open Folder...** -> Enter `C:\BACKTEST`.
4. Open any script (e.g. `run_symbol_alpha_discovery.py`).
5. Set breakpoints, run Python interactively, and inspect arrays using the VM's Python interpreter and hardware.

---

### WORKFLOW 4: The Cloud Multiplier — Dynamic VM Hardware Scaling
The default VM size is 2 vCPUs (calibrated for 24/7 low-cost live paper trading). When you have a massive computation that requires maximum speed (e.g., millions of parameter combinations or all 10 symbols concurrently):

#### Scale Up (Takes ~60 seconds):
Using Azure CLI on your local PC:
```bash
# Scale to 8 high-performance vCPUs (32 GB RAM):
az vm resize --resource-group TRD-RG --name mt5-web --size Standard_D8s_v5

# Or scale to 16 Compute-Optimized vCPUs (32 GB RAM, 3.5 GHz Turbo):
az vm resize --resource-group TRD-RG --name mt5-web --size Standard_F16s_v2
```
*(Alternatively, go to Azure Portal -> Virtual Machines -> mt5-web -> Size -> Pick D8s_v5 / F16s_v2 -> Click Resize).*

#### Blast Through Computation:
Run your discovery suite with `--max-workers 8` or `--max-workers 16`:
```powershell
python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURUSD --max-workers 16"
```
The 5-year multi-million tick discovery that took hours locally finishes in **5 to 10 minutes**!

#### Scale Back Down to Save Costs:
Once your discovery batch finishes, resize the VM back to baseline B2s or D2s:
```bash
az vm resize --resource-group TRD-RG --name mt5-web --size Standard_B2s
```
**Cost Impact**: Running an F16s_v2 instance for 1 hour costs ~$0.75, giving you high-end workstation compute on demand for pocket change!

---

## 🛡️ Resource Isolation: Protecting the Live Trading Fleet (`H1-ONLINE`)

Because the live forward paper-trading fleet (`H1-ONLINE`, Port 5001) also runs on this Azure VM, high-speed backtesting must coexist safely without causing MT5 tick starvation:

1. **Process Priority Isolation**:
   All remote backtesting jobs launched via `remote_compute_dispatcher.py` automatically set their Windows process priority to `BELOW_NORMAL`. This guarantees that if MT5 receives a new price tick or candle, the operating system instantly prioritizes `mt5_live_runner.py` over the backtest.
2. **Worker Thread Clamping**:
   When running on the baseline 2 vCPU VM, keep `--max-workers 2` (or omit the flag). Never set workers higher than the physical logical processor count.
3. **Zero-Copy Memory-Mapped Ticks**:
   The `tick_engine` utilizes `mmap_mode='r'` (`load_ticks(mmap=True)`). Memory mapping loads tick data directly from the OS file cache on demand, preventing backtests from consuming more than 200–300 MB of RAM.

---

## 📋 Quick Command Cheat Sheet

| Task | Command |
|---|---|
| **One-Time VM Setup** | `ssh azure-vm "pip install numba optuna duckdb polars ta"` |
| **Sync Code to VM** | `scp -r tick_engine data_engineering_suite *.py azure-vm:C:/BACKTEST/` |
| **Sync Symbol Ticks** | `scp -r D:\...\BACKTEST\data\EURUSD\* azure-vm:C:/BACKTEST/data/EURUSD/` |
| **1-Click Dispatch** | `python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURUSD"` |
| **Check Remote CPU/RAM** | `ssh azure-vm "Get-Process python \| Select-Object Id, CPU, WorkingSet64"` |
| **Tail Remote Log** | `ssh azure-vm "Get-Content C:\BACKTEST\job.log -Tail 30"` |
| **Kill Remote Backtest** | `ssh azure-vm "Stop-Process -Name python -Force"` *(Note: restart H1-ONLINE after if needed)* |
| **Pull Results to Local** | `scp -r azure-vm:C:/BACKTEST/results/* D:\...\BACKTEST\results\` |

---
*Created for TRD Quantitative Research & Alpha Engineering Suite.*

