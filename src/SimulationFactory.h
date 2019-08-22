/*
 * SimulationFactory.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef SIMULATIONFACTORY_H_
#define SIMULATIONFACTORY_H_

#include <memory>
#include <iosfwd>

#include "random_walker/MoveGenerator.h"
#include "random_walker/MoveFilter.h"
#include "move_filter/image_move_filter/ImageBoundaryConditions.h"
#include "random_walker/RandomWalker.h"
#include "Parameters.h"

class SimulationFactory {
private:
    std::random_device randomSeed;
    std::unique_ptr<MoveGenerator> moveGenerator;
    std::unique_ptr<ImageBoundaryConditions> imageBC;
    std::unique_ptr<MoveFilter> moveFilter;
    std::unique_ptr<RandomWalker> randomWalker;

    std::unique_ptr<MoveGenerator> createMoveGenerator(const Parameters& parameters);
    std::unique_ptr<MoveFilter> createMoveFilter(const Parameters& parameters, std::ostream& logger);
    std::unique_ptr<MoveFilter> createImageMoveFilter(const Parameters& parameters,
                                                      std::istringstream& moveFilterStream, std::ostream& logger);
    std::unique_ptr<ImageBoundaryConditions> createImageBoundaryConditions(std::istringstream& moveFilterStream);

public:
    SimulationFactory(const Parameters &parameters, std::ostream &logger);

    RandomWalker &getRandomWalker();
};

#endif /* SIMULATIONFACTORY_H_ */
