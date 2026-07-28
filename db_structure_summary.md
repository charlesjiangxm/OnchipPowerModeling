# Database Structure & Shape Summary

**Source DB:** `/scratch/PI/eeweiz/jjiangan/c906_db_net_1cyc_20260729`

**Sample DB (first 2 rows):** `/scratch/PI/eeweiz/jjiangan/OnchipPowerModelingNew/c906_db_net_1cyc_20260729_2rows`

**Total .pkl files:** 5

## Original DB Structure

The database is organized as a hierarchical directory tree rooted at `c906_db_net_1cyc_20260729/`. Each `.pkl` file is a pickled **pandas DataFrame** (pickle protocol 5, no compression). The DataFrame index is named `time_ns` (simulation time in nanoseconds); each column is a per-signal switching / power feature.

Directory levels:

1. **`aq_core/`** - top-level core aggregate; contains one `.pkl` per benchmark.
2. **Module level** (`aq_core/<module>/`, e.g. `cp0`, `idu`, `ifu`, `iu`, `lsu`, `rtu`, `vidu`, `vpu`) - per-RTL-module aggregate; contains one `.pkl` per benchmark.
3. **Sub-module level** (`aq_core/<module>/<x_aq_*>/`) - per-RTL-sub-module; contains one `.pkl` per benchmark. Columns here are a subset of the parent module's signals.

Each directory holds up to 12 benchmark DataFrames:

| Benchmark pkl | Description |
|---|---|
| `cache_func.pkl` | Cache test benchmark |
| `conv_softmax_func.pkl` | Conv + softmax benchmark |
| `coremark_func.pkl` | CoreMark benchmark |
| `csr_func.pkl` | CSR / privileged test |
| `debug_func.pkl` | Debug-mode test |
| `exception_func.pkl` | Exception handling test |
| `interrupt_func.pkl` | Interrupt test |
| `ISA_FP_func.pkl` | Floating-point ISA test |
| `ISA_INT_func.pkl` | Integer ISA test |
| `ISA_LS_func.pkl` | Load/Store ISA test |
| `ISA_THEAD_func.pkl` | THEAD-extended ISA test |
| `MMU_func.pkl` | MMU test |

## Shape of Every .pkl (original DB)

Shapes are `(rows, cols)`. `rows` = number of simulation cycles (time_ns samples); `cols` = number of signal features in that RTL scope. The sample DB keeps the first 2 rows of each file with identical columns and index.

| # | File Path | Rows | Cols | Size (MB) |
|---:|---|---:|---:|---:|
| 1 | aq_core/ifu/x_aq_ifu_ctrl/csr_func.pkl | 2736 | 20 | 0.1 |
| 2 | aq_core/iu/x_aq_iu_addr_gen/csr_func.pkl | 2736 | 15 | 0.1 |
| 3 | aq_core/rtu/x_aq_rtu_int/MMU_func.pkl | 4233 | 7 | 0.1 |
| 4 | aq_core/rtu/x_aq_rtu_int/csr_func.pkl | 2736 | 7 | 0.1 |
| 5 | aq_core/rtu/x_aq_rtu_int/interrupt_func.pkl | 3563 | 7 | 0.1 |

## Aggregate Statistics

- Directories containing `.pkl` files: 3
- Files successfully processed: 5
- Distinct row counts: [2736, 3563, 4233]
- Sum of columns across all files: 56
