/*
 * GPUSimulationFactory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include "GPUSimulationFactory.h"

GPUSimulationFactory::GPUSimulationFactory(const Parameters& parameters, std::ostream& logger) {
    Move drift = {parameters.driftX, parameters.driftY};

    this->randomWalker.reset(new GPURandomWalker(parameters.numberOfWalks, parameters.numberOfSteps,
                                                 parameters.tracerRadius, drift, nullptr, nullptr));
}

RandomWalker& GPUSimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
