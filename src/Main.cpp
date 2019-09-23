/*
 ============================================================================
 Name        : obstructed_tracer.cpp
 Author      : Piotr Kubala
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

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

        Analyzer analyzer(parameters);
        Analyzer::Result result = analyzer.analyze(msdData);
        std::cout << "[analyze] Results: D = " << result.D << ", α = " << result.alpha << ", R² = " << result.R2;
        std::cout << std::endl;

        return EXIT_SUCCESS;
    }
}

int main(int argc, char **argv){
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

