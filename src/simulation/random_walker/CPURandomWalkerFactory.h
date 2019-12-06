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
#include "simulation/MoveGenerator.h"
#include "simulation/MoveFilter.h"
#include "CPURandomWalker.h"

/**
 * @brief A class which construct CPURandomWalker from given parameters.
 *
 * Before creating the random walker itself, it creates CPU versions of MoveFilter and MoveGenerator based on
 * parameters and hands them to the walker. It also take the responsibility of clearing the memory.
 */
class CPURandomWalkerFactory : public RandomWalkerFactory {
private:
    std::mt19937 seedGenerator;
    unsigned long numberOfWalksInSeries{};
    std::unique_ptr<MoveGenerator> moveGenerator;
    std::unique_ptr<MoveFilter> moveFilter;
    std::unique_ptr<RandomWalker> randomWalker;

    std::unique_ptr<MoveGenerator> createMoveGenerator(const std::string &moveGeneratorParameters);
    std::unique_ptr<MoveFilter> createMoveFilter(const std::string &moveFilterParameters, std::ostream &logger);
    std::unique_ptr<MoveFilter> createImageMoveFilter(std::istringstream &moveFilterStream, std::ostream &logger);

public:
    /**
     * @brief Constructs the factory based on passed arguments.
     *
     * MoveGenerator and MoveFilter classes will be created based on WalkerParameters::moveGeneratorParameters and
     * WalkerParameters::moveFilterParameters textual representations. @a seed is used to create byte generator, which
     * then samples two new seeds: for MoveGenerator and MoveFilter (for MoveFilter::randomValidTracer).
     *
     * @param seed the seed which will be used in MoveFilter and MoveGenerator
     * @param walkerParameters the parameters of the walker, MoveFilter and MoveGenerator
     * @param logger the output stream which will be passed to RandomWalker to show info
     */
    CPURandomWalkerFactory(unsigned long seed, const WalkerParameters &walkerParameters, std::ostream &logger);

    RandomWalker &getRandomWalker() override;
};

#endif /* CPURANDOMWALKERFACTORY_H_ */
