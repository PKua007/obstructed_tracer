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


class Simulation {
private:
    Parameters parameters;
    std::string outputFilePrefix;
    std::unique_ptr<RandomWalkerFactory> simulationFactory;
    MSDData msdData;

    void store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                            std::size_t firstTrajectoryIndex, std::ostream &logger) {
       std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
       for (std::size_t i = 0; i < numberOfTrajectories; i++) {
           auto &trajectory = randomWalker.getTrajectory(i);

           std::size_t trajectoryIndex = i + firstTrajectoryIndex;
           std::string trajectoryFilename = outputFilePrefix + "_" + std::to_string(trajectoryIndex) + ".txt";
           std::ofstream trajectoryFile(trajectoryFilename);
           if (!trajectoryFile)
               die("[Simulation::run] Cannot open " + trajectoryFilename + " to store trajectory");

           trajectoryFile << std::fixed << std::setprecision(6);
           trajectory.store(trajectoryFile);

           logger << "[Simulation::run] Trajectory " << trajectoryIndex << " stored to " << trajectoryFilename;
           logger << ". Initial position: " << trajectory.getFirst();
           logger << ", accepted steps: " << trajectory.getNumberOfAcceptedSteps();
           logger << ", final position: " << trajectory.getLast() << std::endl;
       }
   }

public:
    Simulation(Parameters parameters, const std::string &outputFilePrefix, std::ostream &logger)
            : outputFilePrefix{outputFilePrefix}, msdData(parameters.numberOfSteps), parameters(std::move(parameters))
    {
        logger << "[Simulation::run] " << _OMP_MAXTHREADS << " OpenMP threads are available." << std::endl;
        if (this->parameters.device == "cpu")
            this->simulationFactory.reset(new CPURandomWalkerFactory(this->parameters, logger));
        else if (this->parameters.device == "gpu")
            this->simulationFactory.reset(new GPURandomWalkerFactory(this->parameters, logger));
        else
            die("[Simulation::run] Unknown device: " + this->parameters.device);
    }

    void run(std::ostream &logger) {
        RandomWalker &randomWalker = this->simulationFactory->getRandomWalker();
        MSDData msdData(this->parameters.numberOfSteps);
        std::size_t numberOfAllTrajectories = this->parameters.numberOfSeries * this->parameters.numberOfWalksInSeries;

        for (std::size_t i = 0; i < this->parameters.numberOfSeries; i++) {
            std::size_t startTrajectory = i * this->parameters.numberOfWalksInSeries;
            std::size_t endTrajectory = (i + 1) * this->parameters.numberOfWalksInSeries - 1;
            logger << std::endl;
            logger << "[Simulation::run] Series " << i << ": trajectories " << startTrajectory << " - ";
            logger << endTrajectory << std::endl;
            randomWalker.run(logger);

            if (this->parameters.storeTrajectories)
                store_trajectories(randomWalker, this->outputFilePrefix, startTrajectory, std::cout);

            logger << "[Simulation::run] Calculating mean square displacement data... " << std::flush;
            Timer timer;
            timer.start();
            msdData.addTrajectories(randomWalker);
            timer.stop();
            logger << "completed in " << timer.count() << " μs." << std::endl;
        }
        logger << std::endl;
    }

    MSDData &getMSDData() {
        return this->msdData;
    }
};

int main(int argc, char **argv){
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

    std::string outputFilePrefix = argv[2];

    Simulation simulation(parameters, outputFilePrefix, std::cout);
    simulation.run(std::cout);
    MSDData &msdData = simulation.getMSDData();

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        die("[main] Cannot open " + msdFilename + " to store mean square displacement data");
    msdData.store(msdFile);
    std::cout << "[main] Mean square displacement data stored to " + msdFilename << std::endl;

    std::cout << "[main] Run finished." << std::endl;
    return EXIT_SUCCESS;
}

