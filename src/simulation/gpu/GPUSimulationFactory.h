/*
 * GPUSimulationFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPUSIMULATIONFACTORY_H_
#define GPUSIMULATIONFACTORY_H_

#include <memory>

#include "Parameters.h"
#include "SimulationFactory.h"
#include "GPURandomWalker.h"

class GPUSimulationFactory: public SimulationFactory {
private:
    std::unique_ptr<GPURandomWalker> randomWalker;

public:
    GPUSimulationFactory(const Parameters &parameters, std::ostream &logger);

    RandomWalker &getRandomWalker() override;
};

#endif /* GPUSIMULATIONFACTORY_H_ */
