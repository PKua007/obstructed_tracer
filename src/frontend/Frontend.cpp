/*
 * Frontend.cpp
 *
 *  Created on: 2 gru 2019
 *      Author: pkua
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "Frontend.h"
#include "MSDData.h"
#include "Simulation.h"
#include "utils/Utils.h"

#include "analyzer/AnalyzerImpl.h"
#include "simulation/SimulationImpl.h"


Frontend::Frontend(int argc, char** argv) {
    this->command = argv[0];
    if (argc < 3)
        throw std::runtime_error("[main] Usage: " + command + " [mode] [input file] {mode specific arguments}");

    this->mode = argv[1];

    std::string inputFilename = argv[2];
    std::ifstream inputFile(inputFilename);
    if (!inputFile)
        throw std::runtime_error("[main] Cannot open " + inputFilename + " to read parameters");

    this->parameters = Parameters(inputFile);
    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    this->parameters.print(std::cout);
    std::cout << std::endl;

    std::copy(argv + 3, argv + argc, std::back_inserter(this->additionalArguments));
}

/* Mode performing random walk. See main for the description. Returns exit code for main. */
int Frontend::perform_walk() {
    if (this->additionalArguments.empty())
        throw std::runtime_error("[perform_walk] Usage: " + this->command + " perform_walk [input file] [output files prefix]");

    std::string outputFilePrefix = this->additionalArguments[0];

    SimulationImpl simulation(parameters, outputFilePrefix, std::cout);
    simulation.run(std::cout);
    MSDData &msdData = simulation.getMSDData();

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        throw std::runtime_error("[perform_walk] Cannot open " + msdFilename + " to store mean square displacement data");
    msdData.store(msdFile);
    std::cout << "[perform_walk] Mean square displacement data stored to " + msdFilename << std::endl;

    std::cout << "[perform_walk] Run finished." << std::endl;
    return EXIT_SUCCESS;
}

/* Mode analyzing the results. See main for the description. Returns exit code for main. */
int Frontend::analyze() {
    if (this->additionalArguments.empty())
        throw std::runtime_error("[analyze] Usage: " + this->command + " analyze [input file] [msd file]");

    std::string msdFilename = this->additionalArguments[0];
    std::ifstream msdFile(msdFilename);
    if (!msdFile)
        throw std::runtime_error("[analyze] Cannot open " + msdFilename + " to restore mean square displacement data");

    MSDData msdData;
    msdData.restore(msdFile);

    AnalyzerImpl analyzer(parameters, 0.01, 1.);    // For a while hardcoded range [t_max/100, t_max]
    analyzer.analyze(msdData);
    Analyzer::Result rSquare = analyzer.getRSquareResult();
    Analyzer::Result rVariance = analyzer.getRVarianceResult();

    std::cout << "             <r²> : D = " << rSquare.D << ", α = " << rSquare.alpha << ", R² = " << rSquare.R2;
    std::cout << std::endl;
    std::cout << "    var(x)+var(y) : D = " << rVariance.D << ", α = " << rVariance.alpha << ", R² = ";
    std::cout << rVariance.R2 << std::endl;
    std::cout << "  last point corr : " << analyzer.getLastPointCorrelation() << std::endl;
    std::cout << "middle point corr : " << analyzer.getMiddlePointCorrelation() << std::endl;

    return EXIT_SUCCESS;
}

int Frontend::run() {
    if (this->mode == "perform_walk")
        return this->perform_walk();
    else if (this->mode == "analyze")
        return this->analyze();
    else
        throw std::runtime_error("[main] Unknown mode: " + this->mode);
}
