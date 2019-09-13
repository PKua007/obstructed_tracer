#!/bin/bash

if [ $# == 0 ]; then
    echo "A simple script which:"
    echo "1) Searches for filename.nb files in current folder. Assumes, that"
    echo "   the last statement is without ; and produces Graphics object"
    echo "2) Makes from it Mathematica script, which exports the Graphics to"
    echo "   filename.ppm and saves the new file as filename.nb.m"
    echo "3) Executes the MathKernel to draw the graphics"
    echo
fi

if [ $# != 1 ]; then
    echo "Usage: $0 [resulution]"
    exit
fi

resolution=$1

for notebook in $(ls *.nb); do
    echo "Generating image from ${notebook}"

    (echo 'Export["'"${notebook}.ppm"'", ' && (cat "${notebook}" | sed 's/Red/Black/g') && echo ", RasterSize->${resolution}]; Quit[];") > "${notebook}.m"
    MathKernel -noprompt -initfile "${notebook}.m"
done

