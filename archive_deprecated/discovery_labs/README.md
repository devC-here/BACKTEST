# Discovery Labs Directory

This dedicated folder houses the complete collection of **52 systematic Alpha Discovery Labs** and their corresponding holdout validation engines.

---

## 📂 Structure

- **`alpha_discovery_lab_v1.py` through `v45.py`**: Multi-asset quantitative edge exploration labs testing 160+ specialized market setups.
- **`alpha_optimization_lab_v38.py` & `v41.py`**: Return and spread resilience optimization labs.
- **`alpha_discovery_lab_audnzd_*.py`**: Specialized cross-currency mean-reversion, liquidity sweep, and macro structure retest discovery labs.
- **`alpha_discovery_lab_eurgbp_expansion.py`**: Specialized EURGBP cointegration, session VWAP, and Golden Pocket discovery lab.
- **`validate_lab_*.py` & `audit_lab_*.py`**: Quarantined 30% holdout verification and certification scripts.

---

## ⚡ Execution

All labs in this directory support dynamic symbol targeting via the master orchestrator:

```bash
# List all 52 discovery labs:
.venv\Scripts\python.exe run_symbol_discovery_labs.py --list

# Run a specific lab on any symbol:
.venv\Scripts\python.exe run_symbol_discovery_labs.py --symbol <SYMBOL> --lab v33

# Run a batch of labs:
.venv\Scripts\python.exe run_symbol_discovery_labs.py --symbol <SYMBOL> --batch specialists
.venv\Scripts\python.exe run_symbol_discovery_labs.py --symbol <SYMBOL> --batch v1-v10

# Run all 52 labs in sequence:
.venv\Scripts\python.exe run_symbol_discovery_labs.py --symbol <SYMBOL> --batch all
```

---

## 📜 Standard Operating Procedure

For the full institutional 3-tier protocol, see [`SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md`](../SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md).

