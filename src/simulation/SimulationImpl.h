/*
 * SimulationImpl.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef SIMULATIONIMPL_H_
#define SIMULATIONIMPL_H_

#include <memory>

#include "Simulation.h"
#include "Parameters.h"
#include "RandomWalkerFactory.h"
#include "MSDDataImpl.h"

class SimulationImpl : public Simulation {
private:
    Parameters parameters;
    std::string outputFilePrefix;
    std::unique_ptr<RandomWalkerFactory> simulationFactory;
    MSDDataImpl msdData;

    void store_trajectories(const RandomWalker &randomWalker, const std::string &outputFilePrefix,
                            std::size_t firstTrajectoryIndex, std::ostream &logger);

public:
    SimulationImpl(Parameters parameters, const std::string &outputFilePrefix, std::ostream &logger);

    void run(std::ostream &logger) override;
    MSDData &getMSDData() override { return this->msdData; }
};

#endif /* SIMULATIONIMPL_H_ */
