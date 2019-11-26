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

echo '"sigma" "drift t" "drift θ" "<r²> D" "<r²> dD" "<r²> α" "<r²> dα" "<r²> R²" "Σvar D" "Σvar dD" "Σvar α" "Σvar dα" "Σvar R²" "last p. corr" "middle p. corr"'

for file in $(ls $dataFolder) ; do
    if [[ $file =~ $inputPattern ]] ; then
        sigma=${BASH_REMATCH[1]}
        driftR=${BASH_REMATCH[2]}
        driftTheta=${BASH_REMATCH[3]}

        ./obstructed_tracer analyze ${dataFolder}/${file}/input.txt ${dataFolder}/${file}_msd.txt > tmp_analyze_output.txt
        
        if [[ $? -ne 0 ]] ; then
            echo "Analyzis failed. Stdout in tmp_analyze_output.txt"
            exit 1
        fi
	
	r2ResultArray=($(sed -r -n 's/<r²> : D = ([0-9.e+-]+) ± ([0-9.e+-]+), α = ([0-9.e+-]+) ± ([0-9.e+-]+), R² = ([0-9.e+-]+)$/\1 \2 \3 \4 \5/p' < tmp_analyze_output.txt))
	rVarResultArray=($(sed -r -n 's/var\(x\)\+var\(y\) : D = ([0-9.e+-]+) ± ([0-9.e+-]+), α = ([0-9.e+-]+) ± ([0-9.e+-]+), R² = ([0-9.e+-]+)$/\1 \2 \3 \4 \5/p' < tmp_analyze_output.txt))
        lastPointCorr=($(sed -r -n 's/last point corr : ([0-9.e+-]+)$/\1/p' < tmp_analyze_output.txt))
        middlePointCorr=($(sed -r -n 's/middle point corr : ([0-9.e+-]+)$/\1/p' < tmp_analyze_output.txt))

	rm tmp_analyze_output.txt

        r2D=${r2ResultArray[0]}
        r2dD=${r2ResultArray[1]}
        r2alpha=${r2ResultArray[2]}
        r2dAlpha=${r2ResultArray[3]}
        r2R2=${r2ResultArray[4]}
        
	rVarD=${rVarResultArray[0]}
        rVardD=${rVarResultArray[1]}
        rVaralpha=${rVarResultArray[2]}
        rVardAlpha=${rVarResultArray[3]}
        rVarR2=${rVarResultArray[4]}
        
	echo $sigma $driftR $driftTheta $r2D $r2dD $r2alpha $r2dAlpha $r2R2 $rVarD $rVardD $rVaralpha $rVardAlpha $rVarR2 $lastPointCorr $middlePointCorr
    fi
done
