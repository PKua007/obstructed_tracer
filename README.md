## Short description

A simple cuda-gpu simulation of random walk in a crowded geometry. The repository contains metadata used by NSight so it can be easily set up in this IDE.

## Usage

> ./obstructed_tracer [mode] (mode specific arguments)

Mode can be one of:
- perform_walk,
- analyze.

 ### perform_walk
*(mode specific arguments): [input file] [output files prefix]*
 
It reads the parameters from input file and performs random walk. The output mean square displacement data and, if desired, trajectories data will be saved as

> [output files prefix]\_msd.tx
    
and
    
> [output files prefix]\_[simulation index]\_[trajectory_index].txt

### analyze
*(mode specific arguments): [input file] [msd file]*
    
It reads the parameters from [input file] and calculates the diffusion coefficient D and exponent &alpha; for last two orders of &lt;r<sup>2</sup>&gt;(t) and &lt;var(x)+var(y)&gt;(t) from [msd file]. It also computes the correlation of x and y for the last point and for the middle one on the log scale. It assumes that [msd file] file was generated using the same input file as given.

