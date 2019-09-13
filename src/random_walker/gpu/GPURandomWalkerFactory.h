/*
 * GPURandomWalkerFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPURANDOMWALKERFACTORY_H_
#define GPURANDOMWALKERFACTORY_H_

#include <memory>
#include <random>

#include "Parameters.h"
#include "simulation/RandomWalkerFactory.h"
#include "GPURandomWalker.h"
#include "move_filter/image_move_filter/ImageBoundaryConditions.h"

class GPURandomWalkerFactory: public RandomWalkerFactory {
private:
    std::mt19937 seedGenerator;
    MoveGenerator *moveGenerator;
    MoveFilter *moveFilter;
    ImageBoundaryConditions *imageBoundaryConditions;
    std::unique_ptr<GPURandomWalker> randomWalker;

public:
    GPURandomWalkerFactory(unsigned long seed, const Parameters &parameters, std::ostream &logger);
    ~GPURandomWalkerFactory();

    RandomWalker &getRandomWalker() override;
};

#endif /* GPURANDOMWALKERFACTORY_H_ */
