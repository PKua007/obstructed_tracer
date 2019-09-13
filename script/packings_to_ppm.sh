#!/bin/bash

if [ $# == 0 ] ; then
    echo "Simple script which:"
    echo "1) Searches for packing of OrientedShape2_1 particles:"
    echo "   packing_[particle]_[angle]_[sigma]_volume_[index].bin"
    echo "   (...) things are taken from arguments, [...] are regex-matched"
    echo "2) Generates Mathematica files from {each of them}"
    echo "   ./rsa.2.0 wolfram -f (input file) {packing} true (draw margin)"
    echo "3) Creates *ppm files of packings using wolfram_ppm_export.sh"
    echo "4) Renames and moves them to folder:"
    echo "   (particle)__[angle]_[sigma]_(density)__(drift r)_(drift theta) / [index].ppm"
    echo "   This signature contains all important information"
    echo "5) Deletes all intermediate files: *nb, *nb.m"
    echo
fi

if [ $# != 6 ] ; then
    echo "Usage: $0 (density) (drift r) (drift theta) (periodic draw margin) (image resolution) (rsa input file)"
    exit
fi

density=$1
driftR=$2
driftTheta=$3
drawMargin=$4
resolution=$5
inputFile=$6

pattern='^packing_([a-zA-Z0-9_.\-]+)_([0-9.\-]+)_([0-9.]+)_[0-9]+_([0-9]+)\.bin$'

for packing in $(ls *bin) ; do
    if [[ $packing =~ $pattern ]] ; then
        ./rsa.2.0 wolfram -f "$inputFile" "$packing" true "$drawMargin"
    fi
done

./wolfram_ppm_exporter.sh $resolution

for packing in $(ls *bin) ; do
    if [[ $packing =~ $pattern ]] ; then
        particle=${BASH_REMATCH[1]}
        angle=${BASH_REMATCH[2]}
        sigma=${BASH_REMATCH[3]}
        index=${BASH_REMATCH[4]}

        rm "${packing}.nb" "${packing}.nb.m"
        folder="${particle}__${angle}_${sigma}_${density}__${driftR}_${driftTheta}"
        mkdir --parents "${folder}"
        mv "${packing}.nb.ppm" "${folder}/${index}.ppm"
    fi
done
