/*
 * Frontend.tpp
 *
 *  Created on: 2 gru 2019
 *      Author: pkua
 */

#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "MSDData.h"
#include "Simulation.h"
#include "utils/Utils.h"


template <typename ConcreteSimulation, typename ConcreteAnalyzer>
Frontend<ConcreteSimulation, ConcreteAnalyzer>::Frontend(int argc, char** argv, std::ostream &logger) : logger{logger} {
    this->command = argv[0];
    if (argc < 3)
        throw RunException("[Frontend] Usage: " + command + " [mode] [input file] {mode specific arguments}");

    this->mode = argv[1];

    std::string inputFilename = argv[2];
    std::ifstream inputFile(inputFilename);
    if (!inputFile)
        throw RunException("[Frontend] Cannot open " + inputFilename + " to read parameters");

    this->parameters = Parameters(inputFile);
    this->logger << "[Frontend] Parameters loaded from " + inputFilename << ":" << std::endl;
    this->parameters.print(this->logger);
    this->logger << std::endl;

    std::copy(argv + 3, argv + argc, std::back_inserter(this->additionalArguments));
}

/* Mode performing random walk. See main for the description. Returns exit code for main. */
template <typename ConcreteSimulation, typename ConcreteAnalyzer>
void Frontend<ConcreteSimulation, ConcreteAnalyzer>::perform_walk() {
    if (this->additionalArguments.empty()) {
        throw RunException("[Frontend::perform_walk] Usage: " + this->command + " perform_walk [input file] "
                           "[output files prefix]");
    }

    std::string outputFilePrefix = this->additionalArguments[0];

    ConcreteSimulation simulation(parameters, outputFilePrefix, this->logger);
    simulation.run(this->logger);
    MSDData &msdData = simulation.getMSDData();

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile) {
        throw RunException("[Frontend::perform_walk] Cannot open " + msdFilename + " to store mean square displacement "
                           "data");
    }

    msdData.store(msdFile);
    this->logger << "[Frontend::perform_walk] Mean square displacement data stored to " + msdFilename << std::endl;
    this->logger << "[Frontend::perform_walk] Run finished." << std::endl;
}

/* Mode analyzing the results. See main for the description. Returns exit code for main. */
template <typename ConcreteSimulation, typename ConcreteAnalyzer>
void Frontend<ConcreteSimulation, ConcreteAnalyzer>::analyze() {
    if (this->additionalArguments.empty())
        throw RunException("[Frontend::analyze] Usage: " + this->command + " analyze [input file] [msd file]");

    std::string msdFilename = this->additionalArguments[0];
    std::ifstream msdFile(msdFilename);
    if (!msdFile) {
        throw RunException("[Frontend::analyze] Cannot open " + msdFilename + " to restore mean square displacement "
                           "data");
    }

    MSDData msdData;
    msdData.restore(msdFile);

    ConcreteAnalyzer analyzer(parameters, 0.01, 1.);    // For a while hardcoded range [t_max/100, t_max]
    analyzer.analyze(msdData);
    Analyzer::Result rSquare = analyzer.getRSquareResult();
    Analyzer::Result rVariance = analyzer.getRVarianceResult();

    this->logger << "             <r²> : D = " << rSquare.D << ", α = " << rSquare.alpha << ", R² = " << rSquare.R2;
    this->logger << std::endl;
    this->logger << "    var(x)+var(y) : D = " << rVariance.D << ", α = " << rVariance.alpha << ", R² = ";
    this->logger << rVariance.R2 << std::endl;
    this->logger << "  last point corr : " << analyzer.getLastPointCorrelation() << std::endl;
    this->logger << "middle point corr : " << analyzer.getMiddlePointCorrelation() << std::endl;
}

template <typename ConcreteSimulation, typename ConcreteAnalyzer>
void Frontend<ConcreteSimulation, ConcreteAnalyzer>::run() {
    if (this->mode == "perform_walk") {
        this->perform_walk();
    } else if (this->mode == "analyze") {
        this->analyze();
    } else {
        throw RunException("[Frontend::run] Unknown mode: " + this->mode);
    }
}
