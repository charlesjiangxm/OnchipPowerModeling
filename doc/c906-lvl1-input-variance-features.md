# C906 Level-1 Submodules — Variance-Selected Input Features

This document lists features kept after **variance** feature selection on the eight level-1 ``_input`` datasets under ``db/aq_core/<module>/<module>_input/``.

## Method

- Preprocess: drop zero-variance columns on train, non-overlapped window average (`avg_wsize=128`), train/val/test ratio `[0.8, 0.2, 0.0]`, `seed=42`.
- Feature selection: `method=variance` on **pre-zscore** train `X`, `variance_threshold=0.0`, no `top_k` cap (all survivors above threshold, ranked high→low by raw variance).
- Train benches: all `*_func.pkl` except `conv_softmax*`; test benches: `conv_softmax*`. Power labels from `db/pwr/*_pwr.pkl` with `y_label=x_aq_core/Pc(x_aq_<module>_top)`.

---

## 1. x_aq_cp0_top (CP0 — Coprocessor 0 / CSR)

**Dataset:** `db/aq_core/cp0/cp0_input/`  |  **Counts:** 73 in → 47 after drop_zero_var → **47 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `ifu_cp0_icache_read_data[127:0]` | 128 | 1.780452e+73 |
| 2 | `idu_cp0_ex1_src1_data[63:0]` | 64 | 1.020832e+37 |
| 3 | `idu_cp0_ex1_src0_data[63:0]` | 64 | 6.521987e+36 |
| 4 | `dtu_cp0_rdata[63:0]` | 64 | 3.802488e+30 |
| 5 | `mmu_cp0_data[63:0]` | 64 | 2.551002e+18 |
| 6 | `sysio_cp0_apb_base[39:0]` | 40 | 2.033089e+18 |
| 7 | `idu_cp0_ex1_opcode[31:0]` | 32 | 3.580765e+17 |
| 8 | `rtu_cp0_epc[63:0]` | 64 | 2.968291e+15 |
| 9 | `iu_cp0_ex1_cur_pc[39:0]` | 40 | 1.899702e+15 |
| 10 | `rtu_cp0_tval[63:0]` | 64 | 1.424589e+15 |
| 11 | `idu_cp0_ex1_func[19:0]` | 20 | 1.263968e+10 |
| 12 | `hpcp_cp0_data[63:0]` | 64 | 3515.19 |
| 13 | `idu_cp0_ex1_dst0_reg[5:0]` | 6 | 30.9129 |
| 14 | `idu_cp0_ex1_halt_info[21:0]` | 22 | 3.54604 |
| 15 | `dtu_cp0_dcsr_prv[1:0]` | 2 | 0.310431 |
| 16 | `ifu_yy_xx_no_op` | 1 | 0.152375 |
| 17 | `idu_cp0_ex1_length` | 1 | 0.100049 |
| 18 | `idu_cp0_ex1_expt_illegal` | 1 | 0.0735271 |
| 19 | `lsu_cp0_sync_ack` | 1 | 0.0554122 |
| 20 | `lsu_cp0_fence_ack` | 1 | 0.0552284 |
| 21 | `rtu_yy_xx_dbgon` | 1 | 0.0334363 |
| 22 | `idu_cp0_ex1_gateclk_sel` | 1 | 0.0165876 |
| 23 | `idu_cp0_ex1_dp_sel` | 1 | 0.0165876 |
| 24 | `idu_cp0_ex1_sel` | 1 | 0.0165869 |
| 25 | `idu_cp0_ex1_split` | 1 | 8.250738e-04 |
| 26 | `biu_cp0_me_int` | 1 | 3.121622e-04 |
| 27 | `rtu_cp0_fflags[4:0]` | 5 | 2.932798e-04 |
| 28 | `rtu_yy_xx_expt_vec[4:0]` | 5 | 2.538716e-04 |
| 29 | `rtu_cp0_fs_dirty_updt_dp` | 1 | 4.563850e-05 |
| 30 | `rtu_cp0_fs_dirty_updt` | 1 | 4.152164e-05 |
| 31 | `cpurst_b` | 1 | 2.670654e-05 |
| 32 | `rtu_yy_xx_flush` | 1 | 2.372568e-05 |
| 33 | `ifu_cp0_rst_inv_req` | 1 | 2.208165e-05 |
| 34 | `rtu_cp0_vxsat_vld` | 1 | 1.499978e-05 |
| 35 | `rtu_cp0_fflags_updt` | 1 | 1.499978e-05 |
| 36 | `ifu_cp0_warm_up` | 1 | 7.139896e-06 |
| 37 | `mmu_yy_xx_no_op` | 1 | 2.977561e-06 |
| 38 | `mmu_cp0_tlb_inv_done` | 1 | 5.630324e-07 |
| 39 | `lsu_cp0_icc_done` | 1 | 4.675733e-07 |
| 40 | `ifu_cp0_icache_inv_done` | 1 | 2.448936e-07 |
| 41 | `ifu_cp0_bht_inv_done` | 1 | 2.227124e-07 |
| 42 | `idu_cp0_ex1_expt_acc_error` | 1 | 1.787921e-07 |
| 43 | `rtu_yy_xx_expt_int` | 1 | 1.005706e-07 |
| 44 | `rtu_yy_xx_expt_vld` | 1 | 1.004232e-07 |
| 45 | `lsu_cp0_dcache_read_data_vld` | 1 | 4.469803e-08 |
| 46 | `ifu_cp0_icache_read_data_vld` | 1 | 4.469803e-08 |
| 47 | `rtu_cp0_exit_debug` | 1 | 1.117451e-08 |

## 2. x_aq_idu_top (IDU — Instruction Decode Unit)

**Dataset:** `db/aq_core/idu/idu_input/`  |  **Counts:** 60 in → 45 after drop_zero_var → **45 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `rtu_idu_wb1_data[63:0]` | 64 | 2.312309e+37 |
| 2 | `rtu_idu_fwd1_data[63:0]` | 64 | 1.890174e+37 |
| 3 | `rtu_idu_fwd2_data[63:0]` | 64 | 1.451016e+37 |
| 4 | `rtu_idu_wb0_data[63:0]` | 64 | 5.674571e+35 |
| 5 | `rtu_idu_fwd0_data[63:0]` | 64 | 4.345997e+35 |
| 6 | `ifu_idu_id_inst[31:0]` | 32 | 6.933247e+17 |
| 7 | `rtu_idu_fwd2_reg[5:0]` | 6 | 40.7362 |
| 8 | `rtu_idu_wb1_reg[5:0]` | 6 | 40.1487 |
| 9 | `rtu_idu_fwd1_reg[5:0]` | 6 | 30.8747 |
| 10 | `rtu_idu_wb0_reg[5:0]` | 6 | 28.2416 |
| 11 | `rtu_idu_fwd0_reg[5:0]` | 6 | 24.189 |
| 12 | `ifu_idu_id_halt_info[21:0]` | 22 | 3.21636 |
| 13 | `cp0_idu_fs[1:0]` | 2 | 0.572125 |
| 14 | `cp0_idu_frm[2:0]` | 3 | 0.486235 |
| 15 | `cp0_yy_priv_mode[1:0]` | 2 | 0.242288 |
| 16 | `ifu_idu_id_inst_vld` | 1 | 0.160511 |
| 17 | `rtu_idu_pipeline_empty` | 1 | 0.153192 |
| 18 | `rtu_idu_wb0_vld` | 1 | 0.086574 |
| 19 | `rtu_idu_fwd0_vld` | 1 | 0.0764948 |
| 20 | `ifu_idu_id_bht_pred[1:0]` | 2 | 0.0454075 |
| 21 | `rtu_yy_xx_dbgon` | 1 | 0.0334363 |
| 22 | `rtu_idu_fwd2_vld` | 1 | 0.0199127 |
| 23 | `rtu_idu_wb1_vld` | 1 | 0.019911 |
| 24 | `cp0_idu_issue_stall` | 1 | 0.0164887 |
| 25 | `rtu_idu_fwd1_vld` | 1 | 0.011071 |
| 26 | `lsu_idu_global_full` | 1 | 0.00783026 |
| 27 | `lsu_idu_full` | 1 | 0.00783026 |
| 28 | `iu_idu_bju_full` | 1 | 0.00166286 |
| 29 | `iu_idu_bju_global_full` | 1 | 0.0010272 |
| 30 | `iu_yy_xx_cancel` | 1 | 3.715376e-04 |
| 31 | `vidu_idu_fp_full` | 1 | 3.556514e-04 |
| 32 | `rtu_idu_commit` | 1 | 1.115652e-04 |
| 33 | `rtu_idu_commit_for_bju` | 1 | 9.376414e-05 |
| 34 | `rtu_idu_flush_stall` | 1 | 5.559227e-05 |
| 35 | `iu_idu_div_full` | 1 | 4.565799e-05 |
| 36 | `cpurst_b` | 1 | 2.670654e-05 |
| 37 | `rtu_idu_flush_fe` | 1 | 2.377597e-05 |
| 38 | `rtu_idu_flush_wbt` | 1 | 2.372568e-05 |
| 39 | `iu_idu_mult_issue_stall` | 1 | 1.687508e-05 |
| 40 | `cp0_idu_cskyee` | 1 | 1.361661e-05 |
| 41 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 42 | `hpcp_idu_cnt_en` | 1 | 1.226944e-05 |
| 43 | `cp0_idu_ucme` | 1 | 1.226944e-05 |
| 44 | `ifu_idu_warm_up` | 1 | 7.139896e-06 |
| 45 | `ifu_idu_id_expt_acc_error` | 1 | 1.787921e-07 |

## 3. x_aq_ifu_top (IFU — Instruction Fetch Unit)

**Dataset:** `db/aq_core/ifu/ifu_input/`  |  **Counts:** 63 in → 54 after drop_zero_var → **54 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `biu_ifu_rdata[127:0]` | 128 | 5.672362e+75 |
| 2 | `cp0_ifu_icache_inv_addr[63:0]` | 64 | 6.931602e+31 |
| 3 | `dtu_ifu_debug_inst[31:0]` | 32 | 1.780436e+17 |
| 4 | `rtu_ifu_chgflw_pc[39:0]` | 40 | 2.144587e+15 |
| 5 | `iu_ifu_tar_pc[63:0]` | 64 | 2.144587e+15 |
| 6 | `iu_ifu_bht_cur_pc[39:0]` | 40 | 1.899702e+15 |
| 7 | `mmu_ifu_pa[27:0]` | 28 | 1.592286e+07 |
| 8 | `cp0_ifu_icache_read_index[13:0]` | 14 | 4.553879e+04 |
| 9 | `mmu_ifu_prot[4:0]` | 5 | 4.33837 |
| 10 | `dtu_ifu_halt_info0[21:0]` | 22 | 3.21636 |
| 11 | `dtu_ifu_halt_info1[21:0]` | 22 | 3.21636 |
| 12 | `mmu_ifu_pa_vld` | 1 | 0.173181 |
| 13 | `biu_ifu_rid` | 1 | 0.116069 |
| 14 | `cp0_ifu_icache_en` | 1 | 0.0681693 |
| 15 | `iu_ifu_bht_pred[1:0]` | 2 | 0.05657 |
| 16 | `rtu_yy_xx_dbgon` | 1 | 0.0334363 |
| 17 | `rtu_ifu_dbg_mask` | 1 | 0.0334363 |
| 18 | `cp0_ifu_icache_pref_en` | 1 | 0.0312979 |
| 19 | `cp0_ifu_rst_inv_done` | 1 | 0.0308014 |
| 20 | `cp0_ifu_ras_en` | 1 | 0.0307962 |
| 21 | `cp0_ifu_bht_en` | 1 | 0.0307962 |
| 22 | `cp0_ifu_btb_en` | 1 | 0.0307962 |
| 23 | `idu_ifu_id_stall` | 1 | 0.0272107 |
| 24 | `cp0_ifu_bht_inv` | 1 | 0.0254346 |
| 25 | `iu_ifu_br_vld` | 1 | 0.0162337 |
| 26 | `iu_ifu_br_vld_gate` | 1 | 0.0162321 |
| 27 | `biu_ifu_rlast` | 1 | 0.0160925 |
| 28 | `cp0_ifu_icache_inv_req` | 1 | 0.00735967 |
| 29 | `iu_ifu_bht_taken` | 1 | 0.00622869 |
| 30 | `cp0_ifu_icache_read_tag` | 1 | 0.00271872 |
| 31 | `biu_ifu_arready` | 1 | 0.00146797 |
| 32 | `biu_ifu_rvalid` | 1 | 4.221319e-04 |
| 33 | `iu_ifu_tar_pc_vld_gate` | 1 | 3.715376e-04 |
| 34 | `iu_ifu_tar_pc_vld` | 1 | 3.715376e-04 |
| 35 | `iu_ifu_bht_mispred` | 1 | 3.036730e-04 |
| 36 | `iu_ifu_bht_mispred_gate` | 1 | 3.036605e-04 |
| 37 | `iu_ifu_ret_vld_gate` | 1 | 6.649703e-05 |
| 38 | `iu_ifu_ret_vld` | 1 | 6.649075e-05 |
| 39 | `iu_ifu_link_vld_gate` | 1 | 6.542476e-05 |
| 40 | `iu_ifu_link_vld` | 1 | 6.542476e-05 |
| 41 | `cpurst_b` | 1 | 2.670654e-05 |
| 42 | `iu_ifu_pc_mispred` | 1 | 2.393910e-05 |
| 43 | `iu_ifu_pc_mispred_gate` | 1 | 2.393910e-05 |
| 44 | `rtu_ifu_flush_fe` | 1 | 2.377597e-05 |
| 45 | `rtu_ifu_chgflw_vld` | 1 | 1.789897e-05 |
| 46 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 47 | `dtu_ifu_halt_info_vld` | 1 | 1.261072e-05 |
| 48 | `hpcp_ifu_cnt_en` | 1 | 1.226944e-05 |
| 49 | `cp0_ifu_lpmd_req` | 1 | 2.977561e-06 |
| 50 | `cp0_ifu_in_lpmd` | 1 | 5.475508e-07 |
| 51 | `cp0_ifu_icache_read_req` | 1 | 4.022822e-07 |
| 52 | `mmu_ifu_access_fault` | 1 | 4.022822e-07 |
| 53 | `dtu_ifu_debug_inst_vld` | 1 | 2.898516e-07 |
| 54 | `cp0_ifu_btb_clr` | 1 | 1.115609e-07 |

## 4. x_aq_iu_top (IU — Integer Unit: ALU / BJU / MUL / DIV)

**Dataset:** `db/aq_core/iu/iu_input/`  |  **Counts:** 57 in → 52 after drop_zero_var → **52 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `da_xx_fwd_data[63:0]` | 64 | 2.312309e+37 |
| 2 | `idu_iu_ex1_src2_data[63:0]` | 64 | 1.602902e+37 |
| 3 | `lsu_iu_ex2_data[63:0]` | 64 | 1.451016e+37 |
| 4 | `idu_iu_ex1_src1_data[63:0]` | 64 | 1.020832e+37 |
| 5 | `idu_iu_ex1_src0_data[63:0]` | 64 | 6.521987e+36 |
| 6 | `rtu_iu_ex2_next_pc[39:0]` | 40 | 2.968291e+15 |
| 7 | `rtu_iu_ex2_cur_pc[39:0]` | 40 | 2.678882e+15 |
| 8 | `ifu_iu_chgflw_pc[39:0]` | 40 | 7.420755e+12 |
| 9 | `idu_iu_ex1_func[19:0]` | 20 | 1.263968e+10 |
| 10 | `ifu_iu_ex1_pc_pred[39:0]` | 40 | 3.346163e+08 |
| 11 | `lsu_iu_ex2_dest_reg[4:0]` | 5 | 40.7362 |
| 12 | `da_xx_fwd_dst_reg[5:0]` | 6 | 40.1487 |
| 13 | `idu_iu_ex1_src0_reg[5:0]` | 6 | 37.4628 |
| 14 | `idu_iu_ex1_dst0_reg[5:0]` | 6 | 30.9129 |
| 15 | `idu_iu_ex1_src1_reg[5:0]` | 6 | 17.7301 |
| 16 | `idu_iu_ex1_pipedown_vld` | 1 | 0.154657 |
| 17 | `rtu_iu_ex1_cmplt` | 1 | 0.154656 |
| 18 | `idu_iu_ex1_inst_vld` | 1 | 0.149679 |
| 19 | `rtu_iu_ex1_cmplt_dp` | 1 | 0.145808 |
| 20 | `rtu_iu_ex1_inst_len` | 1 | 0.104668 |
| 21 | `idu_iu_ex1_length` | 1 | 0.100049 |
| 22 | `rtu_iu_div_wb_grant` | 1 | 0.0807531 |
| 23 | `rtu_iu_mul_wb_grant` | 1 | 0.0807369 |
| 24 | `idu_alu_ex1_gateclk_sel` | 1 | 0.0758001 |
| 25 | `idu_iu_ex1_alu_sel` | 1 | 0.0749925 |
| 26 | `idu_iu_ex1_alu_dp_sel` | 1 | 0.0747162 |
| 27 | `mmu_xx_mmu_en` | 1 | 0.0605258 |
| 28 | `idu_iu_ex1_bht_pred[1:0]` | 2 | 0.0522216 |
| 29 | `idu_bju_ex1_gateclk_sel` | 1 | 0.0232425 |
| 30 | `lsu_iu_ex2_data_vld` | 1 | 0.0199127 |
| 31 | `da_xx_fwd_vld` | 1 | 0.019911 |
| 32 | `idu_iu_ex1_bju_br_sel` | 1 | 0.0176086 |
| 33 | `idu_iu_ex1_bju_sel` | 1 | 0.0176086 |
| 34 | `idu_iu_ex1_bju_dp_sel` | 1 | 0.0176051 |
| 35 | `ifu_iu_reset_vld` | 1 | 0.0151807 |
| 36 | `idu_iu_ex1_src0_ready` | 1 | 0.00780845 |
| 37 | `idu_iu_ex1_mult_dp_sel` | 1 | 0.00247873 |
| 38 | `idu_iu_ex1_mult_sel` | 1 | 0.00247873 |
| 39 | `idu_mult_ex1_gateclk_sel` | 1 | 0.00247873 |
| 40 | `idu_iu_ex1_split` | 1 | 8.250738e-04 |
| 41 | `rtu_iu_ex1_inst_split` | 1 | 8.246029e-04 |
| 42 | `idu_iu_ex1_src1_ready` | 1 | 7.098707e-04 |
| 43 | `cpurst_b` | 1 | 2.670654e-05 |
| 44 | `rtu_yy_xx_flush_fe` | 1 | 2.377597e-05 |
| 45 | `rtu_iu_mul_wb_grant_for_full` | 1 | 2.360629e-05 |
| 46 | `ifu_iu_chgflw_vld` | 1 | 1.789897e-05 |
| 47 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 48 | `hpcp_iu_cnt_en` | 1 | 1.226944e-05 |
| 49 | `ifu_iu_warm_up` | 1 | 7.139896e-06 |
| 50 | `idu_iu_ex1_div_sel` | 1 | 5.887022e-06 |
| 51 | `idu_div_ex1_gateclk_sel` | 1 | 5.887022e-06 |
| 52 | `idu_iu_ex1_div_dp_sel` | 1 | 5.887022e-06 |

## 5. x_aq_lsu_top (LSU — Load / Store Unit)

**Dataset:** `db/aq_core/lsu/lsu_input/`  |  **Counts:** 90 in → 72 after drop_zero_var → **72 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `biu_lsu_rdata[127:0]` | 128 | 6.040289e+70 |
| 2 | `vlsu_lsu_fwd_data[63:0]` | 64 | 2.953531e+37 |
| 3 | `idu_lsu_ex1_src2_data[63:0]` | 64 | 1.602902e+37 |
| 4 | `idu_lsu_ex1_src1_data[63:0]` | 64 | 1.020832e+37 |
| 5 | `vlsu_lsu_wdata[63:0]` | 64 | 9.701437e+36 |
| 6 | `vlsu_dtu_data[63:0]` | 64 | 9.701437e+36 |
| 7 | `idu_lsu_ex1_src0_data[63:0]` | 64 | 6.521987e+36 |
| 8 | `cp0_lsu_icc_addr[63:0]` | 64 | 6.931602e+31 |
| 9 | `mmu_lsu_data_req_addr[39:0]` | 40 | 3.445382e+19 |
| 10 | `idu_lsu_ex1_func[19:0]` | 20 | 1.263968e+10 |
| 11 | `mmu_lsu_pa[27:0]` | 28 | 2.475356e+09 |
| 12 | `iu_lsu_ex1_cur_pc[15:0]` | 16 | 3.250652e+08 |
| 13 | `cp0_lsu_dcache_read_idx[16:0]` | 17 | 7.286206e+05 |
| 14 | `idu_lsu_ex1_dst1_reg[5:0]` | 6 | 41.0384 |
| 15 | `idu_lsu_ex1_src2_reg[5:0]` | 6 | 39.7241 |
| 16 | `idu_lsu_ex1_dst0_reg[5:0]` | 6 | 30.9129 |
| 17 | `vlsu_lsu_fwd_dest_reg[4:0]` | 5 | 11.6291 |
| 18 | `idu_lsu_ex1_halt_info[21:0]` | 22 | 3.54604 |
| 19 | `cp0_lsu_mpp[1:0]` | 2 | 0.554881 |
| 20 | `cp0_yy_priv_mode[1:0]` | 2 | 0.242288 |
| 21 | `vlsu_lsu_src2_depd` | 1 | 0.142758 |
| 22 | `idu_lsu_ex1_length` | 1 | 0.100049 |
| 23 | `cp0_lsu_dcache_en` | 1 | 0.0788383 |
| 24 | `mmu_lsu_sec` | 1 | 0.0456151 |
| 25 | `mmu_lsu_ca` | 1 | 0.0454687 |
| 26 | `mmu_lsu_sh` | 1 | 0.0453392 |
| 27 | `mmu_lsu_pa_vld` | 1 | 0.0377259 |
| 28 | `idu_lsu_ex1_gateclk_sel` | 1 | 0.0342347 |
| 29 | `rtu_yy_xx_dbgon` | 1 | 0.0334363 |
| 30 | `idu_lsu_ex1_dp_sel` | 1 | 0.0332204 |
| 31 | `idu_lsu_ex1_sel` | 1 | 0.0329262 |
| 32 | `cp0_lsu_dcache_pref_dist[1:0]` | 2 | 0.0316874 |
| 33 | `cp0_lsu_amr[1:0]` | 2 | 0.0312979 |
| 34 | `cp0_lsu_dcache_pref_en` | 1 | 0.0312979 |
| 35 | `cp0_lsu_dcache_wa` | 1 | 0.0307962 |
| 36 | `biu_lsu_stb_wready` | 1 | 0.016963 |
| 37 | `biu_lsu_vb_wready` | 1 | 0.016868 |
| 38 | `cp0_lsu_icc_op[1:0]` | 2 | 0.0063189 |
| 39 | `cp0_lsu_icc_req` | 1 | 0.00583496 |
| 40 | `biu_lsu_stb_awready` | 1 | 0.0042704 |
| 41 | `biu_lsu_vb_awready` | 1 | 0.00426721 |
| 42 | `biu_lsu_rid[3:0]` | 4 | 0.00307866 |
| 43 | `biu_lsu_arready` | 1 | 0.00145267 |
| 44 | `idu_lsu_ex1_src2_ready` | 1 | 0.00142182 |
| 45 | `idu_lsu_ex1_split` | 1 | 8.250738e-04 |
| 46 | `mmu_lsu_data_req` | 1 | 6.997239e-04 |
| 47 | `vlsu_lsu_src2_reg[4:0]` | 5 | 5.498284e-04 |
| 48 | `cp0_lsu_fence_req` | 1 | 4.656940e-04 |
| 49 | `mmu_lsu_so` | 1 | 4.326982e-04 |
| 50 | `biu_lsu_no_op` | 1 | 2.721785e-04 |
| 51 | `cp0_lsu_dcache_read_type` | 1 | 2.540425e-04 |
| 52 | `biu_lsu_rvalid` | 1 | 3.509631e-05 |
| 53 | `vlsu_lsu_fwd_vld` | 1 | 3.373466e-05 |
| 54 | `biu_lsu_rlast` | 1 | 3.040410e-05 |
| 55 | `cpurst_b` | 1 | 2.670654e-05 |
| 56 | `rtu_yy_xx_flush` | 1 | 2.372568e-05 |
| 57 | `mmu_lsu_buf` | 1 | 1.281384e-05 |
| 58 | `vlsu_xx_no_op` | 1 | 1.226944e-05 |
| 59 | `cp0_lsu_mm` | 1 | 1.226944e-05 |
| 60 | `hpcp_lsu_cnt_en` | 1 | 1.226944e-05 |
| 61 | `dtu_lsu_halt_info_vld` | 1 | 1.226944e-05 |
| 62 | `cp0_lsu_icc_type[1:0]` | 2 | 8.146215e-06 |
| 63 | `ifu_lsu_warm_up` | 1 | 7.139896e-06 |
| 64 | `cp0_lsu_sync_req` | 1 | 2.999726e-06 |
| 65 | `cp0_lsu_mprv` | 1 | 2.910673e-06 |
| 66 | `vlsu_dtu_data_vld_gate` | 1 | 1.663814e-06 |
| 67 | `vlsu_lsu_data_vld` | 1 | 1.663814e-06 |
| 68 | `cp0_lsu_dcache_read_req` | 1 | 1.352115e-06 |
| 69 | `vlsu_dtu_data_vld` | 1 | 7.151684e-07 |
| 70 | `rtu_lsu_expt_exit` | 1 | 1.561770e-07 |
| 71 | `rtu_lsu_expt_ack` | 1 | 1.004232e-07 |
| 72 | `mmu_lsu_access_fault` | 1 | 2.234492e-08 |

## 6. x_aq_rtu_top (RTU — Retire / Trap Unit)

**Dataset:** `db/aq_core/rtu/rtu_input/`  |  **Counts:** 120 in → 98 after drop_zero_var → **98 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `lsu_rtu_wb_data[63:0]` | 64 | 2.312309e+37 |
| 2 | `iu_rtu_ex3_mul_data[63:0]` | 64 | 1.890174e+37 |
| 3 | `lsu_rtu_ex2_data[63:0]` | 64 | 1.451016e+37 |
| 4 | `lsu_rtu_ex1_data[63:0]` | 64 | 1.225181e+37 |
| 5 | `iu_rtu_div_data[63:0]` | 64 | 5.344336e+36 |
| 6 | `dtu_rtu_pending_tval[63:0]` | 64 | 3.277757e+36 |
| 7 | `iu_rtu_ex1_alu_data[63:0]` | 64 | 5.165514e+35 |
| 8 | `iu_rtu_ex1_bju_data[63:0]` | 64 | 1.441619e+35 |
| 9 | `vpu_rtu_gpr_wb_data[63:0]` | 64 | 3.608883e+33 |
| 10 | `cp0_rtu_ex1_wb_data[63:0]` | 64 | 7.928132e+31 |
| 11 | `cp0_rtu_ex1_chgflw_pc[39:0]` | 40 | 5.298801e+21 |
| 12 | `lsu_rtu_ex1_expt_tval[39:0]` | 40 | 1.714840e+21 |
| 13 | `cp0_rtu_ex1_expt_tval[39:0]` | 40 | 5.279818e+16 |
| 14 | `iu_rtu_ex1_next_pc[39:0]` | 40 | 2.144587e+15 |
| 15 | `iu_rtu_ex1_cur_pc[39:0]` | 40 | 1.899702e+15 |
| 16 | `iu_rtu_depd_lsu_next_pc[39:0]` | 40 | 8.397366e+07 |
| 17 | `dtu_rtu_dpc[63:0]` | 64 | 1.029935e+05 |
| 18 | `cp0_rtu_ex1_halt_info[21:0]` | 22 | 4348.33 |
| 19 | `cp0_rtu_trap_pc[39:0]` | 40 | 2711.49 |
| 20 | `iu_rtu_div_preg[5:0]` | 6 | 174.571 |
| 21 | `lsu_rtu_ex1_dest_reg[5:0]` | 6 | 41.0165 |
| 22 | `lsu_rtu_ex2_dest_reg[5:0]` | 6 | 40.7362 |
| 23 | `lsu_rtu_wb_dest_reg[5:0]` | 6 | 40.1487 |
| 24 | `iu_rtu_ex1_bju_preg[5:0]` | 6 | 30.9129 |
| 25 | `iu_rtu_ex1_alu_preg[5:0]` | 6 | 30.9129 |
| 26 | `iu_rtu_ex3_mul_preg[5:0]` | 6 | 30.8747 |
| 27 | `lsu_rtu_ex1_halt_info[21:0]` | 22 | 3.54604 |
| 28 | `cp0_rtu_int_vld[14:0]` | 15 | 1.68729 |
| 29 | `cp0_rtu_ex1_expt_vec[4:0]` | 5 | 0.294142 |
| 30 | `iu_xx_no_op` | 1 | 0.15091 |
| 31 | `iu_rtu_ex1_bju_inst_len` | 1 | 0.100808 |
| 32 | `iu_rtu_ex1_alu_inst_len` | 1 | 0.100049 |
| 33 | `cp0_rtu_ex1_inst_len` | 1 | 0.100049 |
| 34 | `lsu_rtu_ex1_inst_len` | 1 | 0.0994714 |
| 35 | `lsu_rtu_no_op` | 1 | 0.0763933 |
| 36 | `iu_rtu_ex1_alu_cmplt` | 1 | 0.0749925 |
| 37 | `iu_rtu_ex1_alu_wb_vld` | 1 | 0.0749925 |
| 38 | `iu_rtu_ex1_alu_wb_dp` | 1 | 0.0747162 |
| 39 | `iu_rtu_ex1_alu_cmplt_dp` | 1 | 0.0747162 |
| 40 | `mmu_xx_mmu_en` | 1 | 0.0605258 |
| 41 | `lsu_rtu_ex1_cmplt_dp` | 1 | 0.0380461 |
| 42 | `lsu_rtu_ex1_cmplt` | 1 | 0.0329262 |
| 43 | `lsu_rtu_ex1_cmplt_for_pcgen` | 1 | 0.0329262 |
| 44 | `cp0_rtu_fence_idle` | 1 | 0.0308014 |
| 45 | `iu_rtu_ex1_branch_inst` | 1 | 0.0278896 |
| 46 | `lsu_rtu_ex2_data_vld` | 1 | 0.0199127 |
| 47 | `lsu_rtu_wb_vld` | 1 | 0.019911 |
| 48 | `iu_rtu_ex1_bju_cmplt_dp` | 1 | 0.0183205 |
| 49 | `iu_rtu_ex1_bju_cmplt` | 1 | 0.0176087 |
| 50 | `cp0_rtu_ex1_cmplt_dp` | 1 | 0.0165876 |
| 51 | `cp0_rtu_ex1_flush` | 1 | 0.015337 |
| 52 | `lsu_rtu_ex1_vs_dirty` | 1 | 0.0133173 |
| 53 | `cp0_rtu_ex1_wb_dp` | 1 | 0.0121478 |
| 54 | `iu_rtu_ex3_mul_wb_vld` | 1 | 0.011071 |
| 55 | `lsu_rtu_ex1_buffer_vld` | 1 | 0.00783026 |
| 56 | `vpu_rtu_ex1_fp_dirty` | 1 | 0.00527236 |
| 57 | `lsu_rtu_ex1_fs_dirty` | 1 | 0.00484518 |
| 58 | `iu_rtu_ex1_mul_cmplt_dp` | 1 | 0.00245359 |
| 59 | `iu_rtu_ex1_mul_cmplt` | 1 | 0.00245359 |
| 60 | `vpu_rtu_gpr_wb_index[5:0]` | 6 | 0.00168425 |
| 61 | `vpu_rtu_no_op` | 1 | 9.241571e-04 |
| 62 | `lsu_rtu_ex1_inst_split` | 1 | 8.343149e-04 |
| 63 | `iu_rtu_ex1_alu_inst_split` | 1 | 8.250738e-04 |
| 64 | `cp0_rtu_ex1_inst_split` | 1 | 8.250738e-04 |
| 65 | `lsu_rtu_ex1_wb_dp` | 1 | 3.850067e-04 |
| 66 | `vidu_rtu_no_op` | 1 | 3.673544e-04 |
| 67 | `lsu_rtu_ex1_wb_vld` | 1 | 3.583822e-04 |
| 68 | `cp0_rtu_ex1_wb_preg[5:0]` | 6 | 3.161992e-04 |
| 69 | `vpu_rtu_fflag[5:0]` | 6 | 2.932798e-04 |
| 70 | `iu_rtu_ex1_bju_wb_dp` | 1 | 1.555703e-04 |
| 71 | `iu_rtu_ex1_bju_wb_vld` | 1 | 1.553503e-04 |
| 72 | `vpu_rtu_ex1_cmplt_dp` | 1 | 8.255039e-05 |
| 73 | `vpu_rtu_ex1_cmplt` | 1 | 7.439810e-05 |
| 74 | `cp0_rtu_ex1_cmplt` | 1 | 5.389074e-05 |
| 75 | `cp0_rtu_ex1_wb_vld` | 1 | 4.998309e-05 |
| 76 | `cpurst_b` | 1 | 2.670654e-05 |
| 77 | `vpu_rtu_gpr_wb_req` | 1 | 2.522761e-05 |
| 78 | `vpu_rtu_fflag_vld` | 1 | 1.499978e-05 |
| 79 | `lsu_rtu_ex1_tval2_vld` | 1 | 1.354846e-05 |
| 80 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 81 | `hpcp_rtu_cnt_en` | 1 | 1.226944e-05 |
| 82 | `iu_rtu_div_wb_vld` | 1 | 1.141564e-05 |
| 83 | `iu_rtu_div_wb_dp` | 1 | 1.141564e-05 |
| 84 | `ifu_rtu_warm_up` | 1 | 7.139896e-06 |
| 85 | `iu_rtu_depd_lsu_chgflow_vld` | 1 | 6.769878e-06 |
| 86 | `iu_rtu_ex1_div_cmplt` | 1 | 5.887022e-06 |
| 87 | `iu_rtu_ex1_div_cmplt_dp` | 1 | 5.887022e-06 |
| 88 | `vpu_rtu_ex1_vec_dirty` | 1 | 4.528828e-06 |
| 89 | `lsu_rtu_ex1_expt_vec[4:0]` | 5 | 8.267702e-07 |
| 90 | `cp0_rtu_in_lpmd` | 1 | 5.475508e-07 |
| 91 | `cp0_rtu_ex1_expt_vld` | 1 | 2.121069e-07 |
| 92 | `cp0_rtu_ex1_chgflw` | 1 | 1.561770e-07 |
| 93 | `cp0_rtu_ex1_inst_mret` | 1 | 1.226944e-07 |
| 94 | `iu_rtu_ex2_bju_ras_mispred` | 1 | 6.698564e-08 |
| 95 | `cp0_rtu_ex1_inst_ebreak` | 1 | 3.351124e-08 |
| 96 | `lsu_rtu_ex1_expt_vld` | 1 | 2.234492e-08 |
| 97 | `dtu_rtu_resume_req` | 1 | 1.117451e-08 |
| 98 | `cp0_rtu_ex1_inst_sret` | 1 | 1.117451e-08 |

## 7. x_aq_vidu_top (VIDU — Vector / FP Instruction Decode Unit)

**Dataset:** `db/aq_core/vidu/vidu_input/`  |  **Counts:** 26 in → 19 after drop_zero_var → **19 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `idu_vidu_ex1_inst_data[179:0]` | 180 | 2.699819e+106 |
| 2 | `vpu_vidu_fp_wb_data[63:0]` | 64 | 2.953531e+37 |
| 3 | `vpu_vidu_fp_fwd_data[63:0]` | 64 | 7.993316e+33 |
| 4 | `vpu_vidu_wbt_fp_wb1_reg[4:0]` | 5 | 40.1487 |
| 5 | `vpu_vidu_fp_wb_reg[4:0]` | 5 | 11.6291 |
| 6 | `vpu_vidu_wbt_fp_wb0_reg[4:0]` | 5 | 11.6291 |
| 7 | `vpu_vidu_fp_fwd_reg[4:0]` | 5 | 0.00326676 |
| 8 | `idu_vidu_ex1_fp_gateclk_sel` | 1 | 6.316358e-04 |
| 9 | `idu_vidu_ex1_fp_dp_sel` | 1 | 8.743713e-05 |
| 10 | `idu_vidu_ex1_fp_sel` | 1 | 7.930209e-05 |
| 11 | `vpu_vidu_fp_wb_vld` | 1 | 3.373466e-05 |
| 12 | `vpu_vidu_fp_fwd_vld` | 1 | 3.363407e-05 |
| 13 | `vpu_vidu_wbt_fp_wb0_vld` | 1 | 3.208805e-05 |
| 14 | `cpurst_b` | 1 | 2.670654e-05 |
| 15 | `rtu_vidu_flush_wbt` | 1 | 2.372568e-05 |
| 16 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 17 | `ifu_vidu_warm_up` | 1 | 7.139896e-06 |
| 18 | `vpu_vidu_wbt_fp_wb1_vld` | 1 | 9.590878e-07 |
| 19 | `vpu_vidu_vex1_fp_stall` | 1 | 1.117451e-08 |

## 8. x_aq_vpu_top (VPU — Vector / FP Processing Unit)

**Dataset:** `db/aq_core/vpu/vpu_input/`  |  **Counts:** 51 in → 33 after drop_zero_var → **33 kept**

| Rank | Feature | Width | Variance |
|------|---------|-------|----------|
| 1 | `lsu_vlsu_data[63:0]` | 64 | 2.312309e+37 |
| 2 | `vidu_vpu_vid_fp_inst_srcf0_data[63:0]` | 64 | 1.266380e+37 |
| 3 | `vidu_vpu_vid_fp_inst_srcf1_data[63:0]` | 64 | 1.260438e+37 |
| 4 | `vidu_vpu_vid_fp_inst_src1_data[63:0]` | 64 | 1.029618e+37 |
| 5 | `vidu_vpu_vid_fp_inst_srcf2_data[63:0]` | 64 | 9.054152e+36 |
| 6 | `vidu_vpu_vid_fp_inst_func[19:0]` | 20 | 1.263516e+10 |
| 7 | `lsu_vlsu_func[19:0]` | 20 | 2.378911e+09 |
| 8 | `vidu_vpu_vid_fp_inst_eu[9:0]` | 10 | 1431.63 |
| 9 | `lsu_vlsu_dest_reg[4:0]` | 5 | 40.1487 |
| 10 | `vidu_vpu_vid_fp_inst_dst_reg[5:0]` | 6 | 30.8249 |
| 11 | `vidu_vpu_vid_fp_inst_dstf_reg[4:0]` | 5 | 29.8767 |
| 12 | `lsu_vlsu_st_size[1:0]` | 2 | 1.16001 |
| 13 | `lsu_vlsu_dc_nf[2:0]` | 3 | 0.552893 |
| 14 | `cp0_vpu_xx_rm[2:0]` | 3 | 0.486235 |
| 15 | `lsu_vlsu_st_sew[1:0]` | 2 | 0.39909 |
| 16 | `rtu_vpu_gpr_wb_grnt` | 1 | 0.086778 |
| 17 | `vidu_vpu_vid_fp_inst_dst_vld` | 1 | 0.07314 |
| 18 | `lsu_vlsu_dc_stall` | 1 | 0.0160162 |
| 19 | `vidu_vpu_vid_fp_inst_dstf_vld` | 1 | 0.00534244 |
| 20 | `vidu_vpu_vid_fp_inst_srcf2_vld` | 1 | 0.00337235 |
| 21 | `vidu_vpu_vid_fp_inst_dste_vld` | 1 | 0.00320346 |
| 22 | `cp0_vpu_xx_dqnan` | 1 | 0.00257672 |
| 23 | `vidu_vpu_vid_fp_inst_gateclk_vld` | 1 | 0.00111564 |
| 24 | `vidu_vpu_vid_fp_inst_dp_vld` | 1 | 5.918427e-04 |
| 25 | `vidu_vpu_vid_fp_inst_vld` | 1 | 7.928533e-05 |
| 26 | `cpurst_b` | 1 | 2.670654e-05 |
| 27 | `rtu_yy_xx_flush` | 1 | 2.372568e-05 |
| 28 | `vidu_vpu_vid_fp_inst_srcf2_rdy` | 1 | 1.414323e-05 |
| 29 | `cp0_yy_clk_en` | 1 | 1.281384e-05 |
| 30 | `ifu_vpu_warm_up` | 1 | 7.139896e-06 |
| 31 | `lsu_vlsu_dc_fld_req` | 1 | 1.360265e-06 |
| 32 | `lsu_vlsu_data_vld` | 1 | 9.590878e-07 |
| 33 | `lsu_vlsu_data_grant` | 1 | 3.235286e-07 |
