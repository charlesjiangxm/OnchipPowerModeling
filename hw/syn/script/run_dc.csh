#!/bin/tcsh -f

# Run Synopsys Design Compiler synthesis for numerical_feature_tokenizer.
# Usage:
#   ./run_dc.csh -mode syn [-batch_dir <directory>] [parameter overrides]
#   ./run_dc.csh --clean
#
# Optional overrides:
#   -clock_period <ns>
#   -input_delay <ns>
#   -output_delay <ns>
#   -clock_uncertainty <ns>
#   -output_load <cap>
#   -n_feature <count>
#   -d_token <count>
#   -data_width <bits>
#   -frac_bits <bits>

if ( $#argv == 0 ) then
    echo "Error: No argument provided."
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    echo "       $0 --clean"
    exit 1
endif

set mode = ""
set batch_dir = ""
set clean = 0

set script_dir = `dirname "$0"`
cd "$script_dir"
set script_dir = `pwd`
set syn_dir = `dirname "$script_dir"`
cd "$syn_dir"

while ( $#argv > 0 )
    switch ( "$1" )
        case "--clean":
            set clean = 1
            shift
            breaksw
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
        case "-n_feature":
            if ( $#argv < 2 ) then
                echo "Error: -n_feature requires an argument"
                exit 1
            endif
            setenv N_FEATURE "$2"
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
        default:
            echo "Error: Unknown option '$1'"
            echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
            echo "       $0 --clean"
            exit 1
    endsw
end

if ( "$clean" == "1" ) then
    echo "Cleaning generated synthesis files under ${syn_dir}"
    set nonomatch
    foreach path ( batch_* batch_layer_norm_* alib-* dc.log dc_layer_norm.log command.log view_command.log default.svf )
        if ( -e "$path" ) then
            echo "Removing ${path}"
            /usr/bin/rm -rf "$path"
        endif
    end
    unset nonomatch
    exit 0
endif

if ( "$mode" == "" ) then
    echo "Error: -mode is required"
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    echo "       $0 --clean"
    exit 1
endif

if ( "$mode" != "syn" ) then
    echo "Error: Invalid mode '$mode'. Must be 'syn'"
    echo "Usage: $0 -mode syn [-batch_dir <directory>] [parameter overrides]"
    echo "       $0 --clean"
    exit 1
endif

if ( "$batch_dir" != "" ) then
    setenv BATCH_DIR "$batch_dir"
endif

set tcl_script = "${script_dir}/dc.tcl"
set log_file   = "dc.log"

echo "Running DC synthesis with ${tcl_script}, log saved to ${syn_dir}/${log_file}"
dc_shell -f ${tcl_script} |& tee -i ${log_file}
