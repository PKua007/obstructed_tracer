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

#include "simulation/RandomWalkerFactory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"
#include "move_filter/image_move_filter/ImageBoundaryConditions.h"
#include "CPURandomWalker.h"

class CPURandomWalkerFactory : public RandomWalkerFactory {
private:
    std::mt19937 seedGenerator;
    unsigned long numberOfWalksInSeries{};
    std::unique_ptr<MoveGenerator> moveGenerator;
    std::unique_ptr<ImageBoundaryConditions> imageBC;
    std::unique_ptr<MoveFilter> moveFilter;
    std::unique_ptr<RandomWalker> randomWalker;

    std::unique_ptr<MoveGenerator> createMoveGenerator(const std::string &moveGeneratorParameters);
    std::unique_ptr<MoveFilter> createMoveFilter(const std::string &moveFilterParameters, std::ostream &logger);
    std::unique_ptr<MoveFilter> createImageMoveFilter(std::istringstream &moveFilterStream, std::ostream &logger);
    std::unique_ptr<ImageBoundaryConditions> createImageBoundaryConditions(std::istringstream &moveFilterStream);

public:
    CPURandomWalkerFactory(unsigned long seed, const std::string &moveGeneratorParameters,
                           const std::string &moveFilterParameters, std::size_t numberOfWalksInSeries,
                           const RandomWalker::WalkParameters &walkParameters, std::ostream &logger);

    RandomWalker &getRandomWalker() override;
};

#endif /* CPURANDOMWALKERFACTORY_H_ */
