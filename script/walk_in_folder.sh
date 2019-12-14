#!/bin/bash

if [ $# == 0 ] ; then
    echo "A simple script, which:"
    echo "1) Searches the given folder for *.ppm files"
    echo "   The folder pattern:"
    echo "   <anything>_[drift r]_[drift theta]"
    echo "   Stuff in [...] will be regex matched for further use"
    echo "2) Prepares input file for obstructed_tracer with ImageMoveFilters"
    echo "   using those images"
    echo "3) Prints the comment which should be used to run simulations."
    echo "   The output data will be stored in ./ with prefix the same as"
    echo "   folder name"
    echo
    echo "Input file will be created from tracer_input_pattern.txt. The"
    echo "MoveFilter will be pasted where 'move_filter_placeholder' phrase is"
    echo "appears. Moreover, previously regexed [drift r] and [drift theta]"
    echo "will be pastes into 'drift_r_placeholder' and"
    echo "'drift_theta_placeholder'"
    echo
fi

if [[ $# < 1 || $# > 2 ]] ; then 
    echo "Usage: $0 [folder] (tracer input file pattern)"
    exit
fi

folder=$1
tracerInputFile=$2

if [[ "${tracerInputFile}" == "" ]] ; then
    tracerInputFile=tracer_input_pattern.txt	
fi

pattern='^.*_([0-9.]+)_([0-9.\-]+)$'
if [[ $folder =~ $pattern ]] ; then
    driftR=${BASH_REMATCH[1]}
    driftTheta=${BASH_REMATCH[2]}
else
    echo "Folder ${folder} does not match the desired pattern. Ses $0 (no arguments)"
    exit
fi


moveFilter=""
numberOfMoveFilters=0
for image in $(bash -c "cd $folder && ls *.ppm") ; do
    moveFilter="${moveFilter}ImageMoveFilter ${image} PeriodicBoundaryConditions ; "
    ((numberOfMoveFilters++))
done

cat "${tracerInputFile}" | sed "s/move_filter_placeholder/${moveFilter}/g
                                s/drift_r_placeholder/${driftR}/g
                                s/drift_theta_placeholder/${driftTheta}/g" \
                          > "${folder}/input.txt"

outputPrefix="${folder}"
echo "Prepared ${numberOfMoveFilters} simulations. Run them using command:"
echo "(cd ${folder} && ../obstructed_tracer perform_walk input.txt ../${outputPrefix})"
