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
#include "utils/OMPDefines.h"
#include "random_walker/cpu/CPURandomWalkerFactory.h"
#include "random_walker/gpu/GPURandomWalkerFactory.h"
#include "Timer.h"
#include "MSDData.h"

namespace {
    void store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                            std::size_t firstTrajectoryIndex, std::ostream &logger) {
        std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
        for (std::size_t i = 0; i < numberOfTrajectories; i++) {
            auto &trajectory = randomWalker.getTrajectory(i);

            std::size_t trajectoryIndex = i + firstTrajectoryIndex;
            std::string trajectoryFilename = outputFilePrefix + "_" + std::to_string(trajectoryIndex) + ".txt";
            std::ofstream trajectoryFile(trajectoryFilename);
            if (!trajectoryFile)
                die("[main] Cannot open " + trajectoryFilename + " to store trajectory");

            trajectoryFile << std::fixed << std::setprecision(6);
            trajectory.store(trajectoryFile);

            logger << "[main] Trajectory " << trajectoryIndex << " stored to " << trajectoryFilename;
            logger << ". Initial position: " << trajectory.getFirst();
            logger << ", accepted steps: " << trajectory.getNumberOfAcceptedSteps();
            logger << ", final position: " << trajectory.getLast() << std::endl;
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

    std::cout << "[main] " << _OMP_MAXTHREADS << " OpenMP threads are available." << std::endl;
    std::unique_ptr<RandomWalkerFactory> simulationFactory;
    if (parameters.device == "cpu")
        simulationFactory.reset(new CPURandomWalkerFactory(parameters, std::cout));
    else if (parameters.device == "gpu")
        simulationFactory.reset(new GPURandomWalkerFactory(parameters, std::cout));
    else
        die("[main] Unknown device: " + parameters.device);

    RandomWalker &randomWalker = simulationFactory->getRandomWalker();
    MSDData msdData(parameters.numberOfSteps);
    std::size_t numberOfAllTrajectories = parameters.numberOfSeries * parameters.numberOfWalksInSeries;

    std::string outputFilePrefix = argv[2];
    for (std::size_t i = 0; i < parameters.numberOfSeries; i++) {
        std::size_t startTrajectory = i * parameters.numberOfWalksInSeries;
        std::size_t endTrajectory = (i + 1) * parameters.numberOfWalksInSeries - 1;
        std::cout << std::endl;
        std::cout << "[main] Series " << i << ": trajectories " << startTrajectory << " - " << endTrajectory;
        std::cout << std::endl;
        randomWalker.run(std::cout);

        if (parameters.storeTrajectories)
            store_trajectories(randomWalker, outputFilePrefix, startTrajectory, std::cout);

        std::cout << "[main] Calculating mean square displacement data... " << std::flush;
        Timer timer;
        timer.start();
        msdData.addTrajectories(randomWalker);
        timer.stop();
        std::cout << "completed in " << timer.count() << " μs." << std::endl;
    }
    std::cout << std::endl;

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        die("[main] Cannot open " + msdFilename + " to store mean square displacement data");
    msdData.store(msdFile);
    std::cout << "[main] Mean square displacement data stored to " + msdFilename << std::endl;

    std::cout << "[main] Run finished." << std::endl;
    return EXIT_SUCCESS;
}

