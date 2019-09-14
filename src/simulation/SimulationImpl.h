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
#include "MSDDataImpl.h"

class SimulationImpl : public Simulation {
private:
    std::mt19937 seedGenerator;
    Parameters parameters;
    std::vector<std::string> moveFilters;
    RandomWalkerFactory::WalkerParameters walkerParametersTemplate;
    std::string outputFilePrefix;
    MSDDataImpl msdData;

    Move parseDrift(const std::string &driftString) const;
    RandomWalkerFactory::WalkerParameters prepareWalkerParametersTemplate(const Parameters &parameters) const;
    std::vector<std::string> prepareMoveFilterParameters(const std::string &moveFilterChain) const;
    void initializeSeedGenerator(std::string seed, std::ostream &logger);
    void runSingleSimulation(std::size_t simulationIndex, RandomWalker &randomWalker, std::ostream &logger);
    void store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                            std::size_t simulationIndex, std::size_t firstTrajectoryIndex, std::ostream &logger);

public:
    SimulationImpl(const Parameters &parameters, const std::string &outputFilePrefix, std::ostream &logger);

    void run(std::ostream &logger) override;
    MSDData &getMSDData() override { return this->msdData; }
};

#endif /* SIMULATIONIMPL_H_ */
