/*
 * CPURandomWalkerFactory.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef CPURANDOMWALKERFACTORY_H_
#define CPURANDOMWALKERFACTORY_H_

#include <memory>
#include <iosfwd>

#include "Parameters.h"
#include "simulation/RandomWalkerFactory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"
#include "move_filter/image_move_filter/ImageBoundaryConditions.h"
#include "CPURandomWalker.h"

class CPURandomWalkerFactory : public RandomWalkerFactory {
private:
    std::mt19937 seedGenerator;
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
    CPURandomWalkerFactory(unsigned long seed, const Parameters &parameters, std::ostream &logger);

    RandomWalker &getRandomWalker() override;
};

#endif /* CPURANDOMWALKERFACTORY_H_ */
