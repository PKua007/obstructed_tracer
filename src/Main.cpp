/*
 ============================================================================
 Name        : obstructed_tracer.cpp
 Author      : Piotr Kubala
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

/** @file */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

#include "Parameters.h"
#include "MSDData.h"
#include "Simulation.h"
#include "utils/Utils.h"
#include "analyzer/Analyzer.h"

#include "simulation/SimulationImpl.h"

namespace {
    /* Mode performing random walk. See main for the description. Returns exit code for main. */
    int perform_walk(int argc, char **argv, const Parameters &parameters) {
        std::string command = argv[0];
        if (argc < 4)
            die("[perform_walk] Usage: " + command + " perform_walk [input file] [output files prefix]");

        std::string outputFilePrefix = argv[3];

        auto simulation = std::unique_ptr<Simulation>(new SimulationImpl(parameters, outputFilePrefix, std::cout));
        simulation->run(std::cout);
        MSDData &msdData = simulation->getMSDData();

        std::string msdFilename = outputFilePrefix + "_msd.txt";
        std::ofstream msdFile(msdFilename);
        if (!msdFile)
            die("[perform_walk] Cannot open " + msdFilename + " to store mean square displacement data");
        msdData.store(msdFile);
        std::cout << "[perform_walk] Mean square displacement data stored to " + msdFilename << std::endl;

        std::cout << "[perform_walk] Run finished." << std::endl;
        return EXIT_SUCCESS;
    }

    /* Mode analyzing the results. See main for the description. Returns exit code for main. */
    int analyze(int argc, char **argv, const Parameters &parameters) {
        std::string command = argv[0];
        if (argc < 4)
            die("[analyze] Usage: " + command + " analyze [input file] [msd file]");

        std::string msdFilename = argv[3];
        std::ifstream msdFile(msdFilename);
        if (!msdFile)
            die("[analyze] Cannot open " + msdFilename + " to restore mean square displacement data");

        MSDData msdData;
        msdData.restore(msdFile);

        Analyzer analyzer(parameters, 0.01, 1.);    // For a while hardcoded range [t_max/100, t_max]
        analyzer.analyze(msdData);
        Analyzer::Result rSquare = analyzer.getRSquareResult();
        Analyzer::Result rVariance = analyzer.getRVarianceResult();

        std::cout << "[analyze] <r²>: D = " << rSquare.D << ", α = " << rSquare.alpha << ", R² = " << rSquare.R2;
        std::cout << std::endl;
        std::cout << "[analyze] <var(x) + var(y)>: D = " << rVariance.D << ", α = " << rVariance.alpha << ", R² = ";
        std::cout << rVariance.R2 << std::endl;

        return EXIT_SUCCESS;
    }
}

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
 *         for last two orders of [msd file] mean square displacement data. It assumes that this file was generated
 *         using the same input file.
 *     </li>
 * </ul>
 */
int main(int argc, char **argv) {
    std::string command = argv[0];
    if (argc < 3)
        die("[main] Usage: " + command + " [mode] [input file] {mode specific arguments}");

    std::string inputFilename = argv[2];
    std::ifstream inputFile(inputFilename);
    if (!inputFile)
        die("[main] Cannot open " + inputFilename + " to read parameters");

    Parameters parameters(inputFile);
    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    parameters.print(std::cout);
    std::cout << std::endl;

    std::string mode = argv[1];
    if (mode == "perform_walk")
        return perform_walk(argc, argv, parameters);
    else if (mode == "analyze")
        return analyze(argc, argv, parameters);
    else
        die("[main] Unknown mode: " + mode);
}

