#!/bin/bash

if [[ $# == 0 ]] ; then
    echo "A simple script which performes a complete simulation for the"
    echo "parameters given. The output of the simulation lands in the same"
    echo "folder, images in the folder with the name of"
    echo "<particle>_<additional attr>__<angle>_<sigma>_<density>__..."
    echo "...<drift r>_<drift theta>, rsa data in the rsa subolder"
fi

if [[ $# != 9 ]] ; then
    echo "Usage: $0 (particle) (angle) (sigma) (additional attr) (density) (drift r) (drift theta) (periodic draw margin) (image resoultion)"
    exit
fi

particle=$1
angle=$2
sigma=$3
attributes=$4
density=$5
driftR=$6
driftTheta=$7
periodicDrawMargin=$8
imageResolution=$9

if [ "${attributes}" == "" ] ; then
    fullAttributes="${angle} ${sigma}"
    particleInFolderName="${particle}"
else
    fullAttributes="${attributes} ${angle} ${sigma}"
    particleInFolderName=$(echo "${particle} ${attributes}" | sed 's/ /_/g')
fi

folderName="${particleInFolderName}__${angle}_${sigma}_${density}__${driftR}_${driftTheta}"
rm -rf "${folderName}"

packingFileName="packing_${particleInFolderName}_${angle}_${sigma}_*"

cat rsa_input_pattern.txt | sed "s/particle_placeholder/${particle}/g
                                 s/attributes_placeholder/${fullAttributes}/g" \
                          > rsa_input.txt

./rsa.2.0 density -f rsa_input.txt 0.2
./packings_to_ppm.sh $density $driftR $driftTheta $periodicDrawMargin $imageResolution rsa_input.txt

mkdir --parents "${folderName}/rsa"
mv $(bash -c "echo ${packingFileName}") rsa_input.txt "${folderName}/rsa"

walkCommand=$(./walk_in_folder.sh ${folderName} | sed -n 2p)
bash -c "$walkCommand"
