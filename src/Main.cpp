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
#include "utils/Utils.h"
#include "simulation/cpu/CPUSimulationFactory.h"
#include "simulation/gpu/GPUSimulationFactory.h"
#include "MSDData.h"

namespace {
    void storeTrajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                           std::ostream &logger) {
        std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
        for (std::size_t i = 0; i < numberOfTrajectories; i++) {
            auto &trajectory = randomWalker.getTrajectory(i);

            std::string trajectoryFilename = outputFilePrefix + "_" + std::to_string(i) + ".txt";
            std::ofstream trajectoryFile(trajectoryFilename);
            if (!trajectoryFile)
                die("[main] Cannot open " + trajectoryFilename + " to store trajectory");

            trajectoryFile << std::fixed << std::setprecision(6);
            trajectory.store(trajectoryFile);
            logger << "[main] Trajectory " << i << " stored to " << trajectoryFilename << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    std::string command = argv[0];
    if (argc < 3)
        die("[main] Usage: " + command + " [input file] [output file prefix]");

    std::string inputFilename = argv[1];
    std::ifstream inputFile(inputFilename);
    if (!inputFile)
        die("[main] Cannot open " + inputFilename + " to read parameters");

    Parameters parameters(inputFile);
    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    parameters.print(std::cout);
    std::cout << std::endl;

    std::unique_ptr<SimulationFactory> simulationFactory;
    if (parameters.device == "cpu")
        simulationFactory.reset(new CPUSimulationFactory(parameters, std::cout));
    else if (parameters.device == "gpu")
        simulationFactory.reset(new GPUSimulationFactory(parameters, std::cout));
    else
        die("[main] Unknown device: " + parameters.device);

    RandomWalker &randomWalker = simulationFactory->getRandomWalker();
    randomWalker.run(std::cout);

    std::string outputFilePrefix = argv[2];

    std::cout << "[main] Calculating mean square displacement data... " << std::flush;
    MSDData msdData(randomWalker);
    std::cout << "completed." << std::endl;

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        die("[main] Cannot open " + msdFilename + " to store mean square displacement data");
    msdData.store(msdFile);
    std::cout << "[main] Mean square displacement data stored to " + msdFilename << std::endl;

    if (parameters.storeTrajectories)
        storeTrajectories(randomWalker, outputFilePrefix, std::cout);

    std::cout << "[main] Run finished." << std::endl;
    return EXIT_SUCCESS;
}

