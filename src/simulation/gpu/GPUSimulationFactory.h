/*
 * GPUSimulationFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPUSIMULATIONFACTORY_H_
#define GPUSIMULATIONFACTORY_H_

#include <memory>
#include <random>

#include "Parameters.h"
#include "SimulationFactory.h"
#include "GPURandomWalker.h"

class GPUSimulationFactory: public SimulationFactory {
private:
    std::mt19937 seedGenerator;
    MoveGenerator *moveGenerator;
    MoveFilter *moveFilter;
    std::unique_ptr<GPURandomWalker> randomWalker;

public:
    GPUSimulationFactory(const Parameters &parameters, std::ostream &logger);
    ~GPUSimulationFactory();

    RandomWalker &getRandomWalker() override;
};

#endif /* GPUSIMULATIONFACTORY_H_ */
