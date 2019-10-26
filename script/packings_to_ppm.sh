#!/bin/bash

if [ $# == 0 ] ; then
    echo "Simple script which:"
    echo "1) Searches for packing of OrientedShape2_1 particles:"
    echo "       packing_[particle]_[angle]_[sigma]_<volume>_[index].bin"
    echo "    or"
    echo "       packing_[particle]_[angle]_[sigma]_[additional attributes]_..."
    echo "       ..<volume>_[index].bin"
    echo "   <...> things are ignored, [...] are regex-group-matched"
    echo "2) Generates periodic povray files from {each of them}"
    echo "       ./rsa.2.0 povray -f (input file) {packing} true"
    echo "3) Creates *ppm files of packings using wolfram_ppm_export.sh"
    echo "4) Renames and moves them to folder:"
    echo "       [particle]__[angle]_[sigma]__(drift r)_(drift theta)..."
    echo "       ... / [index].ppm"
    echo "       [particle]_[additional attributes]__[angle]_[sigma]__..."
    echo "       ...(drift r)_(drift theta) / [index].ppm"
    echo "   (...) things are taken from arguments."
    echo "   This signature contains all important information"
    echo "5) Deletes all intermediate files: *pov"
    echo
fi

if [ $# != 4 ] ; then
    echo "Usage: $0 (drift r) (drift theta) (image resolution) (rsa input file)"
    exit 1
fi

driftR=$1
driftTheta=$2
resolution=$3
inputFile=$4

pattern='^packing_([a-zA-Z0-9]+)_([0-9.\-]+)_([0-9.]+)(|_.*)_[0-9]+_([0-9]+)\.bin$'

echo "******** Making images: generating *pov files ********"

for packing in $(ls *bin) ; do
    if [[ $packing =~ $pattern ]] ; then
        ./rsa.2.0 povray -f "$inputFile" "$packing" true

        if [[ $? -ne 0 ]] ; then
            echo "Povraying ${packing} failed. Aborting the rest"
            exit 1
        fi
    fi
done

echo "******** Making images: exporting images ********"

for packing in $(ls *bin) ; do
    if [[ $packing =~ $pattern ]] ; then
        povray "$packing.pov" +W${resolution} +H${resolution} +FP8

        if [[ $? -ne 0 ]] ; then
            echo "Exporting ${packing}.ppm failed. Aborting the rest"
            exit 1
        fi
    fi
done

echo "******** Making images: moving images and deleting temporary files ********"

for packing in $(ls *bin) ; do
    if [[ $packing =~ $pattern ]] ; then
        particle="${BASH_REMATCH[1]}${BASH_REMATCH[4]}"
        angle=${BASH_REMATCH[2]}
        sigma=${BASH_REMATCH[3]}
        index=${BASH_REMATCH[5]}

        rm "${packing}.pov"
        folder="${particle}__${angle}_${sigma}__${driftR}_${driftTheta}"
        mkdir --parents "${folder}"
        mv "${packing}.ppm" "${folder}/${index}.ppm"
    fi
done
