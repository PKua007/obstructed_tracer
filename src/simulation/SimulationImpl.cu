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
#include "random_walker/RandomWalkerFactoryImpl.h"
#include "utils/Utils.h"
#include "utils/OMPDefines.h"
#include "utils/Assertions.h"
#include "utils/CudaCheck.h"
#include "utils/FileUtils.h"
#include "Timer.h"


void SimulationImpl::TrajectoryPrinter::print(const Trajectory &trajectory, const std::string &filename) {
    std::ofstream trajectoryFile(filename);
    if (!trajectoryFile)
        die("[SimulationImpl::run] Cannot open " + filename + " to store trajectory");

    trajectoryFile << std::fixed << std::setprecision(6);
    trajectory.store(trajectoryFile);
}

Move SimulationImpl::parseDrift(const std::string &driftString) const {
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
    // walkParameters.numberOfSteps is for a single walk, so if walks are split we have to divide by number of splits
    walkParameters.numberOfSteps = parameters.numberOfSteps / parameters.numberOfSplits;
    walkParameters.tracerRadius = parameters.tracerRadius;
    walkParameters.integrationStep = parameters.integrationStep;

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

void SimulationImpl::initializePositionHistogram(const std::string &stepsString, std::size_t numberOfSteps) {
    auto steps = explode(stepsString, ' ');
    for (const auto &stepStr : steps) {
        std::size_t step = std::stoul(stepStr);
        Assert(step < numberOfSteps + 1);    // + 1 because the starting point is always present
        this->positionHistogramSteps.push_back(step);
    }

    if (!this->positionHistogramSteps.empty()) {
        this->positionHistogram = std::unique_ptr<PositionHistogram>(
            new PositionHistogram(this->positionHistogramSteps)
        );
    }
}


void SimulationImpl::initializeSeedGenerator(const std::string &seed, std::ostream& logger) {
    if (seed == "random") {
        unsigned long randomSeed = std::random_device()();
        this->seedGenerator.seed(randomSeed);
        logger << "[SimulationImpl] Using random seed: " << randomSeed << std::endl;
    } else {
        this->seedGenerator.seed(std::stoul(seed));
    }
}

void SimulationImpl::initializeDevice(const std::string &deviceParameters) {
    std::istringstream deviceStream(deviceParameters);
    std::string deviceName;
    deviceStream >> deviceName;
    ValidateMsg(deviceStream, "Wrong device format: \"cpu\" OR \"gpu\" (heap size in bytes = default)");

    if (deviceName == "cpu") {
        this->device = CPU;
    } else if (deviceName == "gpu") {
        this->device = GPU;

        int heapSizeInBytes;
        deviceStream >> heapSizeInBytes;

        if (deviceStream) {
            Validate(heapSizeInBytes > 0);
            cudaCheck( cudaDeviceSetLimit(cudaLimitMallocHeapSize, static_cast<size_t>(heapSizeInBytes)) );
        } else if (!deviceStream.eof()) {
            throw ValidationException("Wrong device format: \"cpu\" OR \"gpu\" (heap size in bytes = default)");
        }
    }
}

void SimulationImpl::initializeCoverageMapAccumulator(const std::string &coverageMapsSize) {
    if (coverageMapsSize.empty())
        return;

    std::istringstream sizeStream(coverageMapsSize);
    std::size_t width, height;
    sizeStream >> width >> height;
    ValidateMsg(sizeStream, "coverageMapsSize should be empty (not to do the maps) or: [width > 0] [height > 0]");
    Validate(width > 0);
    Validate(height > 0);

    this->coverageMapAccumulator = std::unique_ptr<CoverageMapAccumulator>(new CoverageMapAccumulator(width, height));
}

void SimulationImpl::initializeTAMSDCalculators(const Parameters &parameters) {
    std::istringstream tamsdParamStream(parameters.tamsdMode);
    std::string mode;
    tamsdParamStream >> mode;
    ValidateMsg(tamsdParamStream, "Malformed TA MSD mode parameters, usage: [mode] ...");

    if (mode == "none")
        return;
    else if (mode != "individual" && mode != "powerLaw" && mode != "both")
        throw ValidationException("TA MSD mode should be one of: none, individual, powerLaw, both");

    std::size_t deltaStep;
    tamsdParamStream >> deltaStep;
    ValidateMsg(tamsdParamStream, "Malformed TA MSD mode parameters, usage: " + mode + " [delta step] ...");
    Validate(deltaStep > 0);

    // Max delta corresponds to number of steps / 10, because later it is just a mess, even for free diffusion.
    // Moreover, maxDelta should be divisible by deltaStep
    std::size_t maxDelta = (parameters.numberOfSteps / 10) / deltaStep * deltaStep;
    ValidateMsg(maxDelta > 0, "Too large delta step");

    this->tamsdCalculator = std::unique_ptr<TimeAveragedMSDCalculator>(
        new TimeAveragedMSDCalculator(maxDelta, deltaStep, parameters.integrationStep)
    );

    if (mode == "individual" || mode == "both")
        this->shouldStoreTAMSD = true;

    if (mode == "powerLaw" || mode == "both") {
        double relativeFitStart{}, relativeFitEnd{};
        tamsdParamStream >> relativeFitStart >> relativeFitEnd;
        ValidateMsg(tamsdParamStream, "Malformed TA MSD parameters, usage: " + mode + " [relative fit start] "
                                      "[relative fit end]");
        Validate(relativeFitStart > 0);
        Validate(relativeFitStart < relativeFitEnd);
        Validate(relativeFitEnd <= 1);

        this->tamsdPowerLawAccumulator = std::unique_ptr<TAMSDPowerLawAccumulator>(
            new TAMSDPowerLawAccumulator(relativeFitStart, relativeFitEnd)
        );
    }
}

void SimulationImpl::storeTrajectories(const RandomWalker &randomWalker, std::size_t simulationIndex,
                                       std::size_t firstTrajectoryIndex, std::ostream &logger)
{
   std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
   for (std::size_t i = 0; i < numberOfTrajectories; i++) {
       auto &trajectory = randomWalker.getTrajectory(i);

       std::size_t trajectoryIndex = i + firstTrajectoryIndex;
       std::ostringstream trajectoryFilenameStream;
       trajectoryFilenameStream << this->outputFilePrefix << "_traj_" << simulationIndex << "_" << trajectoryIndex;
       trajectoryFilenameStream << ".txt";
       std::string trajectoryFilename = trajectoryFilenameStream.str();

       this->trajectoryPrinter->print(trajectory, trajectoryFilename);

       logger << "[SimulationImpl::run] Trajectory " << trajectoryIndex << " stored to " << trajectoryFilename;
       logger << ". Initial position: " << trajectory.getFirst();
       logger << ", accepted steps: " << trajectory.getNumberOfAcceptedSteps();
       logger << ", final position: " << trajectory.getLast() << std::endl;
   }
}

void SimulationImpl::storeHistograms(std::ostream &logger) {
    auto fileOstreamProvider = std::unique_ptr<FileOstreamProvider>(new FileOstreamProvider());

    for (std::size_t step : this->positionHistogramSteps) {
        std::string histogramFilename = this->outputFilePrefix + "_hist_" + std::to_string(step) + ".txt";

        fileOstreamProvider->setFileDescription("histogram for step " + std::to_string(step));
        auto histogramFile = fileOstreamProvider->openFile(histogramFilename);

        this->positionHistogram->printForStep(step, *histogramFile);
        logger << "[SimulationImpl::run] Position histogram for step " << step << " stored to " << histogramFilename;
        logger << std::endl;
    }
}

void SimulationImpl::storeCoverageMaps(std::ostream &logger) {
    if (this->coverageMapAccumulator == nullptr)
        return;

    auto fileOstreamProvider = std::unique_ptr<FileOstreamProvider>(new FileOstreamProvider());

    std::string mapFilename = this->outputFilePrefix + "_multi_map.txt";
    fileOstreamProvider->setFileDescription("coverage map");
    auto mapFile = fileOstreamProvider->openFile(mapFilename);
    this->coverageMapAccumulator->getCoverageMap().store(*mapFile);

    std::string singleMapFilename = this->outputFilePrefix + "_single_map.txt";
    fileOstreamProvider->setFileDescription("coverage map counted once");
    auto singleMapFile = fileOstreamProvider->openFile(singleMapFilename);
    this->coverageMapAccumulator->getCoverageMapCountedOnce().store(*singleMapFile);

    logger << "[SimulationImpl::run] Coverage maps stored to " << mapFilename << " and " << singleMapFilename;
    logger << std::endl;
}

void SimulationImpl::storeTAMSD(const TimeAveragedMSD &tamsd, std::size_t simulationIndex, std::size_t trajectoryIndex,
                                std::ostream& logger)
{
    std::ostringstream tamsdFilenameStream;
    tamsdFilenameStream << this->outputFilePrefix << "_tamsd_" << simulationIndex << "_" << trajectoryIndex << ".txt";
    std::string tamsdFilename = tamsdFilenameStream.str();

    auto fileOstreamProvider = std::unique_ptr<FileOstreamProvider>(new FileOstreamProvider());
    fileOstreamProvider->setFileDescription("TA MSD");
    auto tamsdFile = fileOstreamProvider->openFile(tamsdFilename);
    tamsd.store(*tamsdFile);

    logger << "[SimulationImpl::run] TA MSD " << trajectoryIndex << " stored to " << tamsdFilename << std::endl;
}

void SimulationImpl::storeTAMSDData(std::ostream &logger) {
    if (this->tamsdPowerLawAccumulator == nullptr)
        return;

    auto fileOstreamProvider = std::unique_ptr<FileOstreamProvider>(new FileOstreamProvider());

    std::string histogramFilename = this->outputFilePrefix + "_alpha_hist.txt";
    fileOstreamProvider->setFileDescription("TA MSD power law exponent histogram");
    auto histogramFile = fileOstreamProvider->openFile(histogramFilename);
    auto histogram = this->tamsdPowerLawAccumulator->getExponentHistogram();
    std::copy(histogram.begin(), histogram.end(), std::ostream_iterator<double>(*histogramFile, "\n"));

    std::string varianceHistogramFilename = this->outputFilePrefix + "_var_alpha_hist.txt";
    fileOstreamProvider->setFileDescription("TA MSD power law variance variance histogram");
    auto varianceHistogramFile = fileOstreamProvider->openFile(varianceHistogramFilename);
    auto varianceHistogram = this->tamsdPowerLawAccumulator->getVarianceExponentHistogram();
    std::copy(varianceHistogram.begin(), varianceHistogram.end(),
              std::ostream_iterator<double>(*varianceHistogramFile, "\n"));

    std::string meanTamsdFilename = this->outputFilePrefix + "_mean_tamsd.txt";
    fileOstreamProvider->setFileDescription("ensemble averaged TA MSD");
    auto meanTamsdFile = fileOstreamProvider->openFile(meanTamsdFilename);
    TimeAveragedMSD meanTamsd = this->tamsdPowerLawAccumulator->getEnsembleAveragedTAMSD();
    meanTamsd.store(*meanTamsdFile);

    logger << "[SimulationImpl::run] TAMSD alpha histogram stored to " << histogramFilename << std::endl;
    logger << "[SimulationImpl::run] TAMSD variance alpha histogram stored to " << varianceHistogramFilename;
    logger << std::endl;
    logger << "[SimulationImpl::run] Ensemble averaged TAMSD stored to " << meanTamsdFilename << std::endl;
    logger << "[SimulationImpl::run] Ensemble averaged TAMSD alpha: ";
    logger << this->tamsdPowerLawAccumulator->getEnsembleAveragedExponent() << std::endl;
    logger << "[SimulationImpl::run] Ensemble averaged TAMSD variance alpha: ";
    logger << this->tamsdPowerLawAccumulator->getEnsembleAveragedVarianceExponent() << std::endl;
}

SimulationImpl::SimulationImpl(const Parameters &parameters, const std::string &outputFilePrefix,
                               std::ostream &logger)
        : SimulationImpl(parameters, std::unique_ptr<RandomWalkerFactory>(new RandomWalkerFactoryImpl(logger)),
                         std::unique_ptr<TrajectoryPrinter>(new TrajectoryPrinter), outputFilePrefix, logger)
{ }

SimulationImpl::SimulationImpl(const Parameters &parameters, std::unique_ptr<RandomWalkerFactory> randomWalkerFactory,
                               std::unique_ptr<TrajectoryPrinter> trajectoryPrinter,
                               const std::string &outputFilePrefix, std::ostream &logger)
        : outputFilePrefix{outputFilePrefix}, randomWalkerFactory{std::move(randomWalkerFactory)},
          trajectoryPrinter{std::move(trajectoryPrinter)}, parameters{parameters},
          msdDataCalculator(parameters.numberOfSteps), msdData(parameters.numberOfSteps)
{
    ValidateMsg(parameters.numberOfSteps % parameters.numberOfSplits == 0,
                "Cannot split " + std::to_string(parameters.numberOfSteps) + " in "
                + std::to_string(parameters.numberOfSplits) + " equal parts");

    this->walkerParametersTemplate = this->prepareWalkerParametersTemplate(parameters);
    this->moveFilters = this->prepareMoveFilterParameters(parameters.moveFilter);
    Validate(!this->moveFilters.empty());

    this->initializePositionHistogram(parameters.positionHistogramSteps, parameters.numberOfSteps);
    this->initializeCoverageMapAccumulator(parameters.coverageMapsSize);
    this->initializeTAMSDCalculators(parameters);
    this->initializeSeedGenerator(this->parameters.seed, logger);
    this->initializeDevice(this->parameters.device);

    logger << "[SimulationImpl] " << _OMP_MAXTHREADS << " OpenMP threads are available." << std::endl;
    logger << "[SimulationImpl] " << moveFilters.size() << " simulations will be performed using MoveFilters:";
    logger << std::endl;
    std::copy(this->moveFilters.begin(), this->moveFilters.end(), std::ostream_iterator<std::string>(logger, "\n"));

    if (this->positionHistogram != nullptr) {
        logger << "[SimulationImpl] Position histograms will be generated for steps: ";
        std::copy(this->positionHistogramSteps.begin(), this->positionHistogramSteps.end(),
                  std::ostream_iterator<std::size_t>(logger, ", "));
        logger << std::endl;
    }

    if (this->tamsdCalculator != nullptr) {
        logger << "[SimulationImpl] TA MSD will be calculated up to Delta = " << this->tamsdCalculator->getMaxDelta();
        logger << " with a step " << this->tamsdCalculator->getDeltaStep() << std::endl;
    }

    logger << std::endl;
}

void SimulationImpl::runSingleSimulation(std::size_t simulationIndex, RandomWalker &randomWalker,
                                         std::ostream &logger)
{
    for (std::size_t i = 0; i < this->parameters.numberOfSeries; i++) {
        std::size_t startTrajectory = i * this->parameters.numberOfWalksInSeries;
        std::size_t endTrajectory = (i + 1) * this->parameters.numberOfWalksInSeries - 1;
        logger << std::endl;
        logger << "[SimulationImpl::run] Simulation " << simulationIndex << ", series " << i << ": trajectories ";
        logger << startTrajectory << " - " << endTrajectory << std::endl;

        auto initialTracers = randomWalker.getRandomInitialTracersVector();
        randomWalker.run(logger, initialTracers);

        if (this->parameters.storeTrajectories)
            this->storeTrajectories(randomWalker, simulationIndex, startTrajectory, logger);

        Timer timer;

        logger << "[SimulationImpl::run] Calculating mean square displacement data... " << std::flush;
        timer.start();
        this->msdDataCalculator.addTrajectories(randomWalker);
        timer.stop();
        logger << "completed in " << timer.countMicroseconds() << " μs." << std::endl;

        if (this->positionHistogram != nullptr) {
            logger << "[SimulationImpl::run] Calculating histograms... " << std::flush;
            timer.start();
            this->positionHistogram->addTrajectories(randomWalker);
            timer.stop();
            logger << "completed in " << timer.countMicroseconds() << " μs." << std::endl;
        }

        if (this->coverageMapAccumulator != nullptr) {
            logger << "[SimulationImpl::run] Generating coverage maps... " << std::flush;
            timer.start();
            this->coverageMapAccumulator->addTrajectories(randomWalker.getTrajectories());
            timer.stop();
            logger << "completed in " << timer.countMicroseconds() << " μs." << std::endl;
        }

        if (this->tamsdCalculator != nullptr) {
            logger << "[SimulationImpl::run] Calculating TA MSD... " << std::flush;
            timer.start();
            if (this->shouldStoreTAMSD)
                logger << std::endl;

            auto tamsds = this->tamsdCalculator->calculate(randomWalker.getTrajectories());
            for (std::size_t i{}; i < tamsds.size(); i++) {
                const auto &tamsd = tamsds[i];
                if (this->shouldStoreTAMSD)
                    this->storeTAMSD(tamsd, simulationIndex, startTrajectory + i, logger);

                if (this->tamsdPowerLawAccumulator != nullptr)
                    this->tamsdPowerLawAccumulator->addTAMSD(tamsd);
            }
            timer.stop();
            logger << "completed in " << timer.countMicroseconds() << " μs." << std::endl;
        }
    }
    logger << std::endl;
}

RandomWalkerFactory::WalkerParameters
SimulationImpl::getWalkerParametersForSimulation(std::size_t simulationIndex) const {
    Expects(simulationIndex < this->moveFilters.size());

    auto walkerParameters = this->walkerParametersTemplate;
    walkerParameters.moveFilterParameters = this->moveFilters[simulationIndex];
    return walkerParameters;
}

std::size_t SimulationImpl::getNumberOfSimulations() const {
    return this->moveFilters.size();
}


void SimulationImpl::run(std::ostream &logger) {
    for (std::size_t simulationIndex = 0; simulationIndex < this->getNumberOfSimulations(); simulationIndex++) {
        logger << "[SimulationImpl::run] Preparing simulation " << simulationIndex << "..." << std::endl;

        auto walkerParameters = this->getWalkerParametersForSimulation(simulationIndex);

        std::unique_ptr<RandomWalker> randomWalker;
        if (this->device == CPU)
            randomWalker = this->randomWalkerFactory->createCPURandomWalker(this->seedGenerator(), walkerParameters);
        else if (this->device == GPU)
            randomWalker = this->randomWalkerFactory->createGPURandomWalker(this->seedGenerator(), walkerParameters);
        else
            throw std::runtime_error("");

        if (this->parameters.numberOfSplits != 1) {
            randomWalker = this->randomWalkerFactory->createSplitRandomWalker(this->parameters.numberOfSplits,
                                                                              std::move(randomWalker));
        }

        this->runSingleSimulation(simulationIndex, *randomWalker, logger);
    }

    this->storeHistograms(logger);
    this->storeCoverageMaps(logger);
    this->storeTAMSDData(logger);
    this->msdData = this->msdDataCalculator.fetchMSDData();
}
