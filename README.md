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

> [output files prefix]_msd.tx
    
and
    
> [output files prefix]_[simulation index]_[trajectory_index].txt

### analyze
*(mode specific arguments): [input file] [msd file]*
    
It reads the parameters from [input file] and calculates the diffusion coefficient D and exponent α for last two orders of [msd file] mean square displacement data. It assumes that this file was generated using the same input file.

