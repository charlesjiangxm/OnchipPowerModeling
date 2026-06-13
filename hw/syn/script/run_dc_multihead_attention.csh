#!/bin/tcsh -f

# Run Synopsys Design Compiler synthesis for multihead_attention_registered.
# Usage:
#   ./run_dc_multihead_attention.csh -mode syn [-batch_dir <directory>] [overrides]
#
# Optional overrides:
#   -clock_period <ns>
#   -input_delay <ns>
#   -output_delay <ns>
#   -clock_uncertainty <ns>
#   -output_load <cap>
#   -d_token <count>
#   -n_heads <count>
#   -seq_len <count>
#   -data_width <bits>
#   -frac_bits <bits>
#   -scale_frac <bits>
#   -sm_frac <bits>
#   -recip_frac <bits>
#   -scale <int>           (defaults to round(2^scale_frac / sqrt(d_token/n_heads)))

if ( $#argv == 0 ) then
    echo "Error: No argument provided."
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    exit 1
endif

set mode = ""
set batch_dir = ""

set script_dir = `dirname "$0"`
cd "$script_dir"
set script_dir = `pwd`
set syn_dir = `dirname "$script_dir"`
cd "$syn_dir"

while ( $#argv > 0 )
    switch ( "$1" )
        case "-mode":
            if ( $#argv < 2 ) then
                echo "Error: -mode requires an argument"
                exit 1
            endif
            set mode = "$2"
            shift; shift
            breaksw
        case "-batch_dir":
            if ( $#argv < 2 ) then
                echo "Error: -batch_dir requires an argument"
                exit 1
            endif
            set batch_dir = "$2"
            shift; shift
            breaksw
        case "-clock_period":
            if ( $#argv < 2 ) then
                echo "Error: -clock_period requires an argument"
                exit 1
            endif
            setenv CLOCK_PERIOD "$2"
            shift; shift
            breaksw
        case "-input_delay":
            if ( $#argv < 2 ) then
                echo "Error: -input_delay requires an argument"
                exit 1
            endif
            setenv INPUT_DELAY "$2"
            shift; shift
            breaksw
        case "-output_delay":
            if ( $#argv < 2 ) then
                echo "Error: -output_delay requires an argument"
                exit 1
            endif
            setenv OUTPUT_DELAY "$2"
            shift; shift
            breaksw
        case "-clock_uncertainty":
            if ( $#argv < 2 ) then
                echo "Error: -clock_uncertainty requires an argument"
                exit 1
            endif
            setenv CLOCK_UNCERTAINTY "$2"
            shift; shift
            breaksw
        case "-output_load":
            if ( $#argv < 2 ) then
                echo "Error: -output_load requires an argument"
                exit 1
            endif
            setenv OUTPUT_LOAD "$2"
            shift; shift
            breaksw
        case "-d_token":
            if ( $#argv < 2 ) then
                echo "Error: -d_token requires an argument"
                exit 1
            endif
            setenv D_TOKEN "$2"
            shift; shift
            breaksw
        case "-n_heads":
            if ( $#argv < 2 ) then
                echo "Error: -n_heads requires an argument"
                exit 1
            endif
            setenv N_HEADS "$2"
            shift; shift
            breaksw
        case "-seq_len":
            if ( $#argv < 2 ) then
                echo "Error: -seq_len requires an argument"
                exit 1
            endif
            setenv SEQ_LEN "$2"
            shift; shift
            breaksw
        case "-data_width":
            if ( $#argv < 2 ) then
                echo "Error: -data_width requires an argument"
                exit 1
            endif
            setenv DATA_WIDTH "$2"
            shift; shift
            breaksw
        case "-frac_bits":
            if ( $#argv < 2 ) then
                echo "Error: -frac_bits requires an argument"
                exit 1
            endif
            setenv FRAC_BITS "$2"
            shift; shift
            breaksw
        case "-scale_frac":
            if ( $#argv < 2 ) then
                echo "Error: -scale_frac requires an argument"
                exit 1
            endif
            setenv SCALE_FRAC "$2"
            shift; shift
            breaksw
        case "-sm_frac":
            if ( $#argv < 2 ) then
                echo "Error: -sm_frac requires an argument"
                exit 1
            endif
            setenv SM_FRAC "$2"
            shift; shift
            breaksw
        case "-recip_frac":
            if ( $#argv < 2 ) then
                echo "Error: -recip_frac requires an argument"
                exit 1
            endif
            setenv RECIP_FRAC "$2"
            shift; shift
            breaksw
        case "-scale":
            if ( $#argv < 2 ) then
                echo "Error: -scale requires an argument"
                exit 1
            endif
            setenv SCALE "$2"
            shift; shift
            breaksw
        default:
            echo "Error: Unknown option '$1'"
            echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
            exit 1
    endsw
end

if ( "$mode" == "" ) then
    echo "Error: -mode is required"
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    exit 1
endif

if ( "$mode" != "syn" ) then
    echo "Error: Invalid mode '$mode'. Must be 'syn'"
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    exit 1
endif

if ( "$batch_dir" != "" ) then
    setenv BATCH_DIR "$batch_dir"
endif

set tcl_script = "${script_dir}/dc_multihead_attention.tcl"
set log_file   = "dc_multihead_attention.log"

echo "Running DC synthesis with ${tcl_script}, log saved to ${syn_dir}/${log_file}"
dc_shell -f ${tcl_script} |& tee -i ${log_file}
