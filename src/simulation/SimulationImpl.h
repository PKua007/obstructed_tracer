/*
 * SimulationImpl.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef SIMULATIONIMPL_H_
#define SIMULATIONIMPL_H_

#include <memory>
#include <random>

#include "frontend/Simulation.h"
#include "frontend/Parameters.h"
#include "RandomWalkerFactory.h"
#include "AccumulatingMSDDataCalculator.h"
#include "PositionHistogram.h"
#include "CoverageMapAccumulator.h"
#include "TimeAveragedMSDCalculator.h"
#include "TAMSDPowerLawAccumulator.h"

/**
 * @brief The concrete implementation of Simulation.
 *
 * <p>Based on parameters, it chooses the GPU or CPU simulation model and performs walks in series. Walks in the series
 * are performed in parallel and after they end, the obtained trajectories are added to resulting MSDData. It takes care
 * of selecting the random generator seed and setting the size of GPU heap.
 *
 * <p>The class will perform the separate series of simulations for each MoveFilter specified in the parameters
 * (see input.txt moveFilter) and everything will be added to the resulting MSDData. In can be for example used to
 * average over different ImageMoveFilter instances.
 */
class SimulationImpl : public Simulation {
public:
    /**
     * @brief A class which prints trajectory to file.
     *
     * It is exctracted for unit testing. Printing method can be overrided.
     */
    class TrajectoryPrinter {
    public:
        virtual ~TrajectoryPrinter() { }

        virtual void print(const Trajectory &trajectory, const std::string &filename);
    };

private:
    enum Device {
        CPU,
        GPU
    };

    Parameters parameters;
    Device device;
    RandomWalkerFactory::WalkerParameters walkerParametersTemplate;
    std::mt19937 seedGenerator;
    std::vector<std::string> moveFilters;
    std::string outputFilePrefix;
    std::vector<std::size_t> positionHistogramSteps;

    std::unique_ptr<RandomWalkerFactory> randomWalkerFactory;
    std::unique_ptr<TrajectoryPrinter> trajectoryPrinter;

    AccumulatingMSDDataCalculator msdDataCalculator;
    std::unique_ptr<PositionHistogram> positionHistogram;
    std::unique_ptr<CoverageMapAccumulator> coverageMapAccumulator;
    std::unique_ptr<TimeAveragedMSDCalculator> tamsdCalculator;
    bool shouldStoreTAMSD{};
    std::unique_ptr<TAMSDPowerLawAccumulator> tamsdPowerLawAccumulator;
    MSDData msdData;

    Move parseDrift(const std::string &driftString) const;
    RandomWalkerFactory::WalkerParameters prepareWalkerParametersTemplate(const Parameters &parameters) const;
    std::vector<std::string> prepareMoveFilterParameters(const std::string &moveFilterChain) const;
    void initializePositionHistogram(const std::string &stepsString, std::size_t numberOfSteps);
    void initializeCoverageMapAccumulator(const std::string &coverageMapsSize);
    void initializeSeedGenerator(const std::string &seed, std::ostream &logger);
    void initializeDevice(const std::string &device);
    void initializeTAMSDCalculators(const Parameters &parameters);
    void runSingleSimulation(std::size_t simulationIndex, RandomWalker &randomWalker, std::ostream &logger);
    void storeTrajectories(const RandomWalker &randomWalker, std::size_t simulationIndex,
                           std::size_t firstTrajectoryIndex, std::ostream &logger);
    void storeTAMSD(const TimeAveragedMSD &tamsd, std::size_t simulationIndex, std::size_t trajectoryIndex,
                    std::ostream &logger);
    void storeHistograms(std::ostream &logger);
    void storeCoverageMaps(std::ostream &logger);
    void storeTAMSDData(std::ostream &logger);

public:
    /**
     * @brief Constructor using the default RandomWalkerFactory - RandomWalkerFactoryImpl and default
     * TrajectoryPrinter.
     *
     * @see SimulationImpl::SimulationImpl(const Parameters &, std::unique_ptr<RandomWalkerFactory>,
     * const std::string &, std::ostream &)
     */
    SimulationImpl(const Parameters &parameters, const std::string &outputFilePrefix, std::ostream &logger);

    /**
     * @brief Constructs the simulation based on @a parameters.
     *
     * It creates the proper - GPU or CPU simulation model. It takes care of selecting the random generator seed and
     * setting the size of GPU heap. The info how @a parameters fields are interpreted is in input.txt file (see
     * Parameters to know how input.txt entries are mapped to Parameters fields).
     *
     * @param parameters parameters of the simulation
     * @param randomWalkerFactory RandomWalkerFactory used to produce concrete RandomWalker-s
     * @param trajectoryPrinter TrajectoryPrinter which prints the trajectory
     * @param outputFilePrefix the prefix of trajectory file name which will be saved if @a parameters want saving
     * trajectories
     * @param logger output stream to show information such as list of MoveFilter parameters for each simulation
     */
    SimulationImpl(const Parameters &parameters, std::unique_ptr<RandomWalkerFactory> randomWalkerFactory,
                   std::unique_ptr<TrajectoryPrinter> trajectoryPrinter, const std::string &outputFilePrefix,
                   std::ostream &logger);

    /**
     * @brief Returns the number of simulations to be performed.
     *
     * The number of simulations is determined by the number of MoveFilter passed in Parameters.
     *
     * @return the number of simulations to be performed
     */
    std::size_t getNumberOfSimulations() const;

    /**
     * @brief Returns random walker parameters prepared from Parameters for simulation of index @a simulationIndex.
     *
     * Parameters for different indices may differ only in MoveFilters.
     *
     * @param simulationIndex index of simulation to prepare parameters for. The number of simulations is determined by
     * the number of MoveFilter passed in Parameters
     * @return random walker parameters prepared from Parameters for simulation of index @a simulationIndex
     */
    RandomWalkerFactory::WalkerParameters getWalkerParametersForSimulation(std::size_t simulationIndex) const;

    /**
     * @brief Performs one or more simulations, based on number of MoveFilter instances used.
     *
     * It will perform the separate series of simulations for each MoveFilter specified in the parameters in the
     * constructor (see input.txt moveFilter) and everything will be added to the resulting MSDData. In can be for
     * example used to average over different ImageMoveFilter instances. The data can then be fetched using getMSDData.
     *
     * @param logger output stream to show info on simulation progress
     */
    void run(std::ostream &logger) override;

    /**
     * @brief Fetches the mean square displacement data calculated from all simulations performed.
     * @return mean square displacement data calculated from all simulations performed
     */
    MSDData &getMSDData() override { return this->msdData; }
};

#endif /* SIMULATIONIMPL_H_ */
