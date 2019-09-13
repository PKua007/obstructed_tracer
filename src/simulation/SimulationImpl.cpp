/*
 * SimulationImpl.cpp
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#include <fstream>
#include <iomanip>

#include "SimulationImpl.h"
#include "utils/Utils.h"
#include "utils/OMPDefines.h"
#include "random_walker/cpu/CPURandomWalkerFactory.h"
#include "random_walker/gpu/GPURandomWalkerFactory.h"
#include "Timer.h"

void SimulationImpl::initializeSeedGenerator(std::string seed, std::ostream& logger) {
    if (seed == "random") {
        unsigned long randomSeed = std::random_device()();
        this->seedGenerator.seed(randomSeed);
        logger << "[SimulationImpl] Using random seed: " << randomSeed << std::endl;
    } else {
        this->seedGenerator.seed(std::stoul(seed));
    }
}

void SimulationImpl::store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                                        std::size_t firstTrajectoryIndex, std::ostream &logger)
{
   std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
   for (std::size_t i = 0; i < numberOfTrajectories; i++) {
       auto &trajectory = randomWalker.getTrajectory(i);

       std::size_t trajectoryIndex = i + firstTrajectoryIndex;
       std::string trajectoryFilename = outputFilePrefix + "_" + std::to_string(trajectoryIndex) + ".txt";
       std::ofstream trajectoryFile(trajectoryFilename);
       if (!trajectoryFile)
           die("[SimulationImpl::run] Cannot open " + trajectoryFilename + " to store trajectory");

       trajectoryFile << std::fixed << std::setprecision(6);
       trajectory.store(trajectoryFile);

       logger << "[SimulationImpl::run] Trajectory " << trajectoryIndex << " stored to " << trajectoryFilename;
       logger << ". Initial position: " << trajectory.getFirst();
       logger << ", accepted steps: " << trajectory.getNumberOfAcceptedSteps();
       logger << ", final position: " << trajectory.getLast() << std::endl;
   }
}

SimulationImpl::SimulationImpl(Parameters parameters, const std::string &outputFilePrefix, std::ostream &logger)
        : outputFilePrefix{outputFilePrefix}, msdData(parameters.numberOfSteps), parameters{parameters}
{
    this->initializeSeedGenerator(this->parameters.seed, logger);

    logger << "[SimulationImpl] " << _OMP_MAXTHREADS << " OpenMP threads are available." << std::endl;

    RandomWalker::WalkParameters walkParameters;
    walkParameters.drift = {parameters.driftX, parameters.driftY};
    walkParameters.numberOfSteps = parameters.numberOfSteps;
    walkParameters.tracerRadius = parameters.tracerRadius;

    if (this->parameters.device == "cpu") {
        this->simulationFactory.reset(new CPURandomWalkerFactory(this->seedGenerator(), parameters.moveGenerator,
                                                                 parameters.moveFilter,
                                                                 parameters.numberOfWalksInSeries, walkParameters,
                                                                 logger));
    } else if (this->parameters.device == "gpu") {
        this->simulationFactory.reset(new GPURandomWalkerFactory(this->seedGenerator(), parameters.moveGenerator,
                                                                 parameters.moveFilter,
                                                                 parameters.numberOfWalksInSeries, walkParameters,
                                                                 logger));
    } else {
        die("[SimulationImpl] Unknown device: " + this->parameters.device);
    }
}

void SimulationImpl::run(std::ostream &logger) {
    RandomWalker &randomWalker = this->simulationFactory->getRandomWalker();
    std::size_t numberOfAllTrajectories = this->parameters.numberOfSeries * this->parameters.numberOfWalksInSeries;

    for (std::size_t i = 0; i < this->parameters.numberOfSeries; i++) {
        std::size_t startTrajectory = i * this->parameters.numberOfWalksInSeries;
        std::size_t endTrajectory = (i + 1) * this->parameters.numberOfWalksInSeries - 1;
        logger << std::endl;
        logger << "[SimulationImpl::run] Series " << i << ": trajectories " << startTrajectory << " - ";
        logger << endTrajectory << std::endl;
        randomWalker.run(logger);

        if (this->parameters.storeTrajectories)
            store_trajectories(randomWalker, this->outputFilePrefix, startTrajectory, logger);

        logger << "[SimulationImpl::run] Calculating mean square displacement data... " << std::flush;
        Timer timer;
        timer.start();
        this->msdData.addTrajectories(randomWalker);
        timer.stop();
        logger << "completed in " << timer.count() << " μs." << std::endl;
    }
    logger << std::endl;
}
