################################################################################
# Design Compiler synthesis for mlp_wrapper
# (3-layer MLP accelerator: gated fc1 -> dyn-quant -> fc2 -> dyn-quant -> fc3
#  -> dyn-quant, int8, round-to-nearest-ties-to-even).
#
# Mirrors dc_feed_forward_network.tcl but targets the MLP block. The RTL uses
# SystemVerilog always_ff/always_comb, so it is analyzed as -format sverilog.
# There is no scalar constant to derive, so only
# N_FEATURES/HIDDEN1/HIDDEN2/DATA_WIDTH are passed to elaborate.
#
# Run:  ./run_dc_mlp.csh -mode syn [-batch_dir <dir>] [param overrides]
################################################################################

################################################################################
# User configuration
################################################################################
set SCRIPT_ROOT     [file normalize [file dirname [info script]]]
set SYN_ROOT        [file normalize [file join ${SCRIPT_ROOT} ..]]
set HW_ROOT         [file normalize [file join ${SYN_ROOT} ..]]
set PROJ_ROOT       [file normalize [file join ${HW_ROOT} ..]]
set RTL_ROOT        ${HW_ROOT}/rtl
set WRAPPER_ROOT    ${SYN_ROOT}/wrapper
set TOP_MODULE_NAME mlp_wrapper

proc get_env_or_default {name default_value} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default_value
}

set N_FEATURES   [get_env_or_default N_FEATURES   32]
set HIDDEN1      [get_env_or_default HIDDEN1      16]
set HIDDEN2      [get_env_or_default HIDDEN2      16]
set DATA_WIDTH   [get_env_or_default DATA_WIDTH   8]
set CLOCK_PERIOD [get_env_or_default CLOCK_PERIOD 1.0]

set INPUT_DELAY       [get_env_or_default INPUT_DELAY       [expr {$CLOCK_PERIOD * 0.20}]]
set OUTPUT_DELAY      [get_env_or_default OUTPUT_DELAY      [expr {$CLOCK_PERIOD * 0.20}]]
set CLOCK_UNCERTAINTY [get_env_or_default CLOCK_UNCERTAINTY [expr {$CLOCK_PERIOD * 0.05}]]
set OUTPUT_LOAD       [get_env_or_default OUTPUT_LOAD       0.01]

# Create a timestamped batch directory for all outputs.  A relative BATCH_DIR is
# interpreted from hw/syn so runs stay contained in this synthesis directory.
if {[info exists ::env(BATCH_DIR)] && $::env(BATCH_DIR) ne ""} {
  set BATCH_DIR $::env(BATCH_DIR)
  if {[file pathtype ${BATCH_DIR}] ne "absolute"} {
    set BATCH_DIR [file normalize [file join ${SYN_ROOT} ${BATCH_DIR}]]
  }
} else {
  set date_hour [clock format [clock seconds] -format "%Y%m%d_%H"]
  set BATCH_DIR [file join ${SYN_ROOT} "batch_mlp_${date_hour}"]
}

file mkdir ${BATCH_DIR}
file mkdir ${BATCH_DIR}/WORK
file mkdir ${BATCH_DIR}/reports
file mkdir ${BATCH_DIR}/results

set param_rpt [open ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.parameters.rpt w]
puts $param_rpt "TOP_MODULE_NAME  ${TOP_MODULE_NAME}"
puts $param_rpt "N_FEATURES       ${N_FEATURES}"
puts $param_rpt "HIDDEN1          ${HIDDEN1}"
puts $param_rpt "HIDDEN2          ${HIDDEN2}"
puts $param_rpt "DATA_WIDTH       ${DATA_WIDTH}"
puts $param_rpt "CLOCK_PERIOD     ${CLOCK_PERIOD}"
puts $param_rpt "INPUT_DELAY      ${INPUT_DELAY}"
puts $param_rpt "OUTPUT_DELAY     ${OUTPUT_DELAY}"
puts $param_rpt "CLOCK_UNCERTAINTY ${CLOCK_UNCERTAINTY}"
puts $param_rpt "OUTPUT_LOAD      ${OUTPUT_LOAD}"
close $param_rpt

################################################################################
# Step 1: library setup
################################################################################
set search_path [list . \
  /dfs/app/tsmc_icdc/tsmc028/28HPCplus_RF/SC/tcbn28hpcplusbwp30p140/tcbn28hpcplusbwp30p140_190a/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140_180a/ \
  /dfs/app/tsmc_icdc/tsmc028/28HPCplus_RF/SC/tcbn28hpcplusbwp30p140hvt/tcbn28hpcplusbwp30p140hvt_190a/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140hvt_180a/ \
  /dfs/app/tsmc_icdc/tsmc028/28HPCplus_RF/SC/tcbn28hpcplusbwp30p140lvt/tcbn28hpcplusbwp30p140lvt_190a/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140lvt_180a/ \
  /dfs/app/tsmc_icdc/tsmc028/28HPCplus_RF/SC/tcbn28hpcplusbwp40p140ehvt/tcbn28hpcplusbwp40p140ehvt_190a/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp40p140ehvt_170a \
]

# Standard-cell libraries, matching the referenced C906 DC flow.
set target_library [list \
  tcbn28hpcplusbwp30p140tt1v25c_ccs.db \
  tcbn28hpcplusbwp30p140hvttt1v25c_ccs.db \
  tcbn28hpcplusbwp30p140lvttt1v25c_ccs.db \
  tcbn28hpcplusbwp40p140ehvttt1v25c_ccs.db \
]

set link_library [concat [list "*"] $target_library]

# Naming rules.
define_name_rules lab_vlog   -type  port  \
        -allowed {a-zA-Z0-9[]_} \
        -equal_ports_nets    \
        -first_restricted  "0-9_"  \
        -max_length   256
define_name_rules lab_vlog   -type  net  \
        -allowed "a-zA-Z0-9_" \
        -equal_ports_nets    \
        -first_restricted  "0-9_"  \
        -max_length   256
define_name_rules lab_vlog   -type  cell  \
        -allowed "a-zA-Z0-9_" \
        -first_restricted  "0-9_"  \
        -map {{{"\[","_","\]",""},{"\[","_"}}}  \
        -max_length   256
define_name_rules slash   -restricted  {/}  -replace  {_}

################################################################################
# Step 2: import design
################################################################################
define_design_lib WORK -path ${BATCH_DIR}/WORK

set rtl_files [list \
  ${RTL_ROOT}/requant_rne.v \
  ${RTL_ROOT}/dyn_quant.v \
  ${RTL_ROOT}/mlp.v \
  ${WRAPPER_ROOT}/mlp_wrapper.v \
]

# RTL uses SystemVerilog always_ff/always_comb -> analyze as sverilog.
analyze -format sverilog $rtl_files

set ELAB_PARAMS "N_FEATURES=${N_FEATURES},HIDDEN1=${HIDDEN1},HIDDEN2=${HIDDEN2},DATA_WIDTH=${DATA_WIDTH}"
elaborate ${TOP_MODULE_NAME} -parameters ${ELAB_PARAMS}
current_design ${TOP_MODULE_NAME}
link
uniquify

# Store the unmapped results.
write -hierarchy -format ddc -output ${BATCH_DIR}/results/${TOP_MODULE_NAME}.unmapped.ddc

################################################################################
# Step 3: constrain design
################################################################################
create_clock -name clk -period ${CLOCK_PERIOD} [get_ports clk]
set_clock_uncertainty ${CLOCK_UNCERTAINTY} [get_clocks clk]

set non_clock_inputs [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_input_delay  ${INPUT_DELAY}  -clock clk ${non_clock_inputs}
set_output_delay ${OUTPUT_DELAY} -clock clk [all_outputs]

set_false_path -from [get_ports rst_n]
set_dont_touch_network [get_ports rst_n]

set MAX_FANOUT     32
set MAX_TRANSITION 0.5
set DRIVING_CELL   "BUFFD2BWP30P140"

set_max_fanout ${MAX_FANOUT} [current_design]
set_max_transition ${MAX_TRANSITION} [current_design]
set_driving_cell -lib_cell ${DRIVING_CELL} ${non_clock_inputs}
set_load ${OUTPUT_LOAD} [all_outputs]

# Create default path groups.
group_path -name REGOUT -to [all_outputs]
group_path -name REGIN -from ${non_clock_inputs}
group_path -name FEEDTHROUGH -from ${non_clock_inputs} -to [all_outputs]

# Prevent assignment statements in the Verilog netlist.
set_fix_multiple_port_nets -all -buffer_constants

# Check for design errors before compile.
check_design -summary
check_design > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.check_design.rpt
check_timing > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.check_timing.precompile.rpt

################################################################################
# Step 4: compile design
################################################################################
compile_ultra -no_autoungroup

################################################################################
# Step 5: write final design and reports
################################################################################
change_names -rules verilog -hierarchy

write -format verilog -hierarchy -output ${BATCH_DIR}/results/${TOP_MODULE_NAME}.mapped.v
write -format ddc -hierarchy -output ${BATCH_DIR}/results/${TOP_MODULE_NAME}.mapped.ddc
write_sdf ${BATCH_DIR}/results/${TOP_MODULE_NAME}.mapped.sdf
write_sdc -nosplit ${BATCH_DIR}/results/${TOP_MODULE_NAME}.mapped.sdc

# PTPX name mapping for downstream power annotation.
saif_map -type ptpx -write_map ${BATCH_DIR}/results/${TOP_MODULE_NAME}.ptpxmap.tcl
report_saif -hier -rtl -missing > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.saif_annotation.rpt

report_qor > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.qor.rpt
report_timing -transition_time -nets -attribute -nosplit \
  > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.timing.rpt
report_area -nosplit > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.area.rpt
report_area -hierarchy -nosplit > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.area_hier.rpt
report_power -hierarchy -nosplit > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.power_hier.rpt
report_reference -hierarchy -nosplit > ${BATCH_DIR}/reports/${TOP_MODULE_NAME}.mapped.reference_hier.rpt

################################################################################
# Exit Design Compiler
################################################################################
exit
