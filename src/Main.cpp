/*
 ============================================================================
 Name        : obstructed_tracer.cpp
 Author      : Piotr Kubala
 ============================================================================
 */

/** @file */

#include <iostream>

#include "frontend/Frontend.h"
#include "simulation/SimulationImpl.h"
#include "analyzer/AnalyzerImpl.h"

/**
 * @brief Entry point with two distinct modes: @a perform_walk and @a analyze
 *
 * Usage:
 * <blockquote>./obstructed_tracer [mode] (mode specific arguments)</blockquote>
 * <ul>
 *     <li>
 *         <p><b>perform_walk</b>
 *         <p><em>arguments: [input file] [output files prefix]</em>
 *         <p>It reads the parameters from [input file] (see Parameters and input.txt) and performs random walk.
 *         The output mean square displacement data and, if desired, trajectories data will be saved as
 *         <blockquote>[output files prefix]_msd.txt</blockquote>
 *         and
 *         <blockquote>[output files prefix]_[simulation index]_[trajectory_index].txt</blockquote>
 *     </li>
 *     <li>
 *         <p><b>analyze</b>
 *         <p><em>arguments: [input file] [msd file]</em>
 *         <p>It reads the parameters from [input file] and calculates the diffusion coefficient D and exponent &alpha;
 *         for last two orders of &lt;r<sup>2</sup>&gt;(t) and &lt;var(x)+var(y)&gt;(t) from [msd file]. It also
 *         computes the correlation of x and y for the last point and for the middle one on the log scale. It assumes
 *         that this file was generated using the same input file as given.
 *     </li>
 * </ul>
 */
int main(int argc, char **argv) {
    Frontend<SimulationImpl, AnalyzerImpl> frontend(argc, argv, std::cout);
    return frontend.run();
}

