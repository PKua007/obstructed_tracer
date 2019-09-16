#!/bin/bash

if [[ $# == 0 ]] ; then
    echo "A simple script which performs a complete simulation for the"
    echo "parameters given. The output of the simulation lands in the same"
    echo "folder, images in the folder with the name of"
    echo "<particle>_<additional attr>__<angle>_<sigma>_<drift r>_..."
    echo "...<drift theta>, rsa data in the rsa subolder"
fi

if [[ $# != 7 ]] ; then
    echo "Usage: $0 (particle) (angle) (sigma) (additional attr) (drift r) (drift theta) (image resoultion)"
    exit 1
fi

particle=$1
angle=$2
sigma=$3
attributes=$4
driftR=$5
driftTheta=$6
imageResolution=$7

if [ "${attributes}" == "" ] ; then
    fullAttributes="${angle} ${sigma}"
    particleInFolderName="${particle}"
else
    fullAttributes="${angle} ${sigma} ${attributes}"
    particleInFolderName=$(echo "${particle} ${attributes}" | sed 's/ /_/g')
fi
particleInPackingFileName=$(echo "${particle} ${fullAttributes}" | sed 's/ /_/g')

folderName="${particleInFolderName}__${angle}_${sigma}__${driftR}_${driftTheta}"
rm -rf "${folderName}"

packingFileName="packing_${particleInPackingFileName}_*"

echo "******** Preparing rsa_input ********"

cat rsa_input_pattern.txt | sed "s/particle_placeholder/${particle}/g
                                 s/attributes_placeholder/${fullAttributes}/g" \
                          > rsa_input.txt

if [[ $? -ne 0 ]] ; then
    echo "Could not prepare rsa_input.txt, aborting packing generation"
    exit 1
fi

echo "******** Generating packing ********"

./rsa.2.0 density -f rsa_input.txt

if [[ $? -ne 0 ]] ; then
    echo "RSA packing generation launch failed, aborting image generation"
    exit 1
fi

echo "******** Generating images of packings ********"

./packings_to_ppm.sh $driftR $driftTheta $imageResolution rsa_input.txt

if [[ $? -ne 0 ]] ; then
    echo "Image generation failed, aborting moving RSA files"
    exit 1
fi

echo "******** Moving RSA files ********"

mkdir --parents "${folderName}/rsa"
mv $(bash -c "echo ${packingFileName}") rsa_input.txt "${folderName}/rsa"

if [[ $? -ne 0 ]] ; then
    echo "Failed to move RSA files, aborting random walk"
    exit 1
fi

echo "******** Starting random walk ********"

walkCommand=$(./walk_in_folder.sh ${folderName} | sed -n 2p)
bash -c "$walkCommand"

if [[ $? -ne 0 ]] ; then
    echo "Random walk failed. It can be redone with:"
    echo "${walkCommand}"
fi
