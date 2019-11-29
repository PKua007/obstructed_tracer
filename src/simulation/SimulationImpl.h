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

#include "Simulation.h"
#include "Parameters.h"
#include "RandomWalkerFactory.h"
#include "AccumulatingMSDDataCalculator.h"

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
private:
    enum Device {
        CPU,
        GPU
    };

    Device device;
    std::mt19937 seedGenerator;
    Parameters parameters;
    std::vector<std::string> moveFilters;
    RandomWalkerFactory::WalkerParameters walkerParametersTemplate;
    std::string outputFilePrefix;
    AccumulatingMSDDataCalculator msdDataCalculator;
    MSDData msdData;

    Move parseDrift(const std::string &driftString) const;
    RandomWalkerFactory::WalkerParameters prepareWalkerParametersTemplate(const Parameters &parameters) const;
    std::vector<std::string> prepareMoveFilterParameters(const std::string &moveFilterChain) const;
    void initializeSeedGenerator(const std::string &seed, std::ostream &logger);
    void initializeDevice(const std::string &device);
    void runSingleSimulation(std::size_t simulationIndex, RandomWalker &randomWalker, std::ostream &logger);
    void store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                            std::size_t simulationIndex, std::size_t firstTrajectoryIndex, std::ostream &logger);

public:
    /**
     * @brief Constructs the simulation based on @a parameters.
     *
     * It creates the proper - GPU or CPU simulation model. It takes care of selecting the random generator seed and
     * setting the size of GPU heap. The info how @a parameters fields are interpreted is in input.txt file (seeParameters to know how input.txt entries are mapped to Parameters fields).
     *
     * @param parameters parameters of the simulation
     * @param outputFilePrefix the prefix of trajectory file name which will be saved if @a parameters want saving
     * trajectories
     * @param logger output stream to show information such as list of MoveFilter parameters for each simulation
     */
    SimulationImpl(const Parameters &parameters, const std::string &outputFilePrefix, std::ostream &logger);

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
     *
     * @return mean square displacement data calculated from all simulations performed
     */
    MSDData &getMSDData() override { return this->msdData; }
};

#endif /* SIMULATIONIMPL_H_ */
