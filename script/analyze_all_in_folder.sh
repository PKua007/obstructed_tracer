if [ $# == 0 ] ; then
    echo "A simple script, which searches given data folder and analyzes all"
    echo "mean square displacement data found there. The names of files and"
    echo "directories structure should be the one of complete_simulaiton.sh,"
    echo "so using this script is a natural continuation of the former one."
    echo
fi

if [ $# != 3 ] ; then
    echo "Usage: $0 [data folder] [particle name] [particle additional attributes]"
    exit 1
fi

dataFolder="$1"
particle="$2"
particleAttributes="$3"

if [ "$particleAttributes" == "" ] ; then
    particleSignature="$particle"
else
    particleSignature="${particle}_$(echo "$particleAttributes" | sed 's/ /_/g')"
fi

inputPattern='^'"$particleSignature"'__[0-9.\-]+_([0-9.\-]+)__([0-9.]+)_([0-9.-]+)$'

for file in $(ls $dataFolder) ; do
    if [[ $file =~ $inputPattern ]] ; then
        sigma=${BASH_REMATCH[1]}
        driftR=${BASH_REMATCH[2]}
        driftTheta=${BASH_REMATCH[3]}

        resultArray=($(./obstructed_tracer analyze ${dataFolder}/${file}/input.txt ${dataFolder}/${file}_msd.txt \
            | sed -r -n 's/^.*D = ([0-9.e+-]+) ± ([0-9.e+-]+), α = ([0-9.e+-]+) ± ([0-9.e+-]+), R² = ([0-9.e+-]+)$/\1 \2 \3 \4 \5/p'))
        D=${resultArray[0]}
        dD=${resultArray[1]}
        alpha=${resultArray[2]}
        dAlpha=${resultArray[3]}
        R2=${resultArray[4]}

        echo $sigma $driftR $driftTheta $D $dD $alpha $dAlpha $R2
    fi
done
