/*
 * SimulationImpl.cpp
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <functional>
#include <iterator>

#include "SimulationImpl.h"
#include "utils/Utils.h"
#include "utils/OMPDefines.h"
#include "utils/Assertions.h"
#include "random_walker/cpu/CPURandomWalkerFactory.h"
#include "random_walker/gpu/GPURandomWalkerFactory.h"
#include "Timer.h"


Move SimulationImpl::parseDrift(const std::string& driftString) const {
    std::istringstream driftStream(driftString);
    std::string coordinatesType;
    float driftCoordinates[2];

    driftStream >> coordinatesType >> driftCoordinates[0] >> driftCoordinates[1];
    ValidateMsg(driftStream, "Drift must be: xy (x) (y) or rt (radius) (angle in degrees)");

    if (coordinatesType == "xy") {
        return {driftCoordinates[0], driftCoordinates[1]};
    } else if (coordinatesType == "rt") {
        float r = driftCoordinates[0];
        float theta = driftCoordinates[1];
        Validate(r >= 0);

        float degreeToRad = float{M_PI} / 180.f;
        return {r * std::cos(theta * degreeToRad), r * std::sin(theta * degreeToRad)};
    } else {
        throw ValidationException("Coordinates type in drift must be 'xy' or 'rt'");
    }
}

RandomWalkerFactory::WalkerParameters
SimulationImpl::prepareWalkerParametersTemplate(const Parameters &parameters) const {
    // Walker parameters can be preprepared, because they are shared between multiple simulations.
    // The simulation can only differ in MoveFilter and its number is determined from Parameters::moveFilter
    RandomWalker::WalkParameters walkParameters;
    walkParameters.drift = this->parseDrift(parameters.drift);
    walkParameters.numberOfSteps = parameters.numberOfSteps;
    walkParameters.tracerRadius = parameters.tracerRadius;

    RandomWalkerFactory::WalkerParameters walkerParametersTemplate;
    walkerParametersTemplate.moveFilterParameters = "";  // placeholder for specific MoveFilters
    walkerParametersTemplate.moveGeneratorParameters = parameters.moveGenerator;
    walkerParametersTemplate.numberOfWalksInSeries = parameters.numberOfWalksInSeries;
    walkerParametersTemplate.walkParameters = walkParameters;

    return walkerParametersTemplate;
}

std::vector<std::string> SimulationImpl::prepareMoveFilterParameters(const std::string &moveFilterChain) const {
    auto moveFilterStrings = explode(moveFilterChain, ';');
    std::for_each(moveFilterStrings.begin(), moveFilterStrings.end(), trim);
    moveFilterStrings.erase(std::remove_if(moveFilterStrings.begin(), moveFilterStrings.end(),
                                           std::mem_fun_ref(&std::string::empty)),
                            moveFilterStrings.end());
    return moveFilterStrings;
}

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
                                        std::size_t simulationIndex, std::size_t firstTrajectoryIndex,
                                        std::ostream &logger)
{
   std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
   for (std::size_t i = 0; i < numberOfTrajectories; i++) {
       auto &trajectory = randomWalker.getTrajectory(i);

       std::size_t trajectoryIndex = i + firstTrajectoryIndex;
       std::ostringstream trajectoryFilenameStream;
       trajectoryFilenameStream << outputFilePrefix << "_" << simulationIndex << "_" << trajectoryIndex << ".txt";
       std::string trajectoryFilename = trajectoryFilenameStream.str();
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

SimulationImpl::SimulationImpl(const Parameters &parameters, const std::string &outputFilePrefix, std::ostream &logger)
        : outputFilePrefix{outputFilePrefix}, msdData(parameters.numberOfSteps), parameters{parameters}
{
    this->walkerParametersTemplate = this->prepareWalkerParametersTemplate(parameters);
    this->moveFilters = this->prepareMoveFilterParameters(parameters.moveFilter);
    Validate(!this->moveFilters.empty());
    this->initializeSeedGenerator(this->parameters.seed, logger);

    logger << "[SimulationImpl] " << _OMP_MAXTHREADS << " OpenMP threads are available." << std::endl;
    logger << "[SimulationImpl] " << moveFilters.size() << " simulations will be performed using MoveFilters:";
    logger << std::endl;
    std::copy(this->moveFilters.begin(), this->moveFilters.end(), std::ostream_iterator<std::string>(logger, "\n"));
    logger << std::endl;
}

void SimulationImpl::runSingleSimulation(std::size_t simulationIndex, RandomWalker &randomWalker,
                                         std::ostream &logger)
{
    std::size_t numberOfAllTrajectories = this->parameters.numberOfSeries * this->parameters.numberOfWalksInSeries;

    for (std::size_t i = 0; i < this->parameters.numberOfSeries; i++) {
        std::size_t startTrajectory = i * this->parameters.numberOfWalksInSeries;
        std::size_t endTrajectory = (i + 1) * this->parameters.numberOfWalksInSeries - 1;
        logger << std::endl;
        logger << "[SimulationImpl::run] Simulation " << simulationIndex << ", series " << i << ": trajectories ";
        logger << startTrajectory << " - " << endTrajectory << std::endl;
        randomWalker.run(logger);

        if (this->parameters.storeTrajectories)
            store_trajectories(randomWalker, this->outputFilePrefix, simulationIndex, startTrajectory, logger);

        logger << "[SimulationImpl::run] Calculating mean square displacement data... " << std::flush;
        Timer timer;
        timer.start();
        this->msdData.addTrajectories(randomWalker);
        timer.stop();
        logger << "completed in " << timer.count() << " μs." << std::endl;
    }
    logger << std::endl;
}

void SimulationImpl::run(std::ostream &logger) {
    for (std::size_t simulationIndex = 0; simulationIndex < this->moveFilters.size(); simulationIndex++) {
        logger << "[SimulationImpl::run] Preparing simulation " << simulationIndex << "..." << std::endl;

        auto walkerParameters = this->walkerParametersTemplate;
        walkerParameters.moveFilterParameters = this->moveFilters[simulationIndex];

        std::unique_ptr<RandomWalkerFactory> randomWalkerFactory;
        if (this->parameters.device == "cpu")
            randomWalkerFactory.reset(new CPURandomWalkerFactory(this->seedGenerator(), walkerParameters, logger));
        else if (this->parameters.device == "gpu")
            randomWalkerFactory.reset(new GPURandomWalkerFactory(this->seedGenerator(), walkerParameters, logger));
        else
            die("[SimulationImpl] Unknown device: " + this->parameters.device);

        this->runSingleSimulation(simulationIndex, randomWalkerFactory->getRandomWalker(), logger);
    }
}
