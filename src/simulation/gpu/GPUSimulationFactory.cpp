/*
 * GPUSimulationFactory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include "GPUSimulationFactory.h"

GPUSimulationFactory::GPUSimulationFactory(const Parameters& parameters, std::ostream& logger) {
    this->randomWalker.reset(new GPURandomWalker());
}

RandomWalker& GPUSimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
