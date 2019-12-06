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
 * parameters and hands them to the walker.
 */
class CPURandomWalkerFactory : public RandomWalkerFactory {
private:
    std::mt19937 seedGenerator;
    WalkerParameters walkerParameters;
    unsigned long numberOfWalksInSeries{};
    std::ostream &logger;

    std::unique_ptr<MoveGenerator> createMoveGenerator(const std::string &moveGeneratorParameters);
    std::unique_ptr<MoveFilter> createMoveFilter(const std::string &moveFilterParameters, std::ostream &logger);
    std::unique_ptr<MoveFilter> createImageMoveFilter(std::istringstream &moveFilterStream, std::ostream &logger);

public:
    /**
     * @brief Constructs the factory based on passed arguments.
     *
     * @a seed is used to create byte generator, which then will samples two new seeds: for MoveGenerator and MoveFilter
     * during creation of CPURandomWalker.
     *
     * @param seed the seed which will be used in MoveFilter and MoveGenerator
     * @param walkerParameters the parameters of the walker, MoveFilter and MoveGenerator
     * @param logger the output stream which will be passed to RandomWalker to show info
     */
    CPURandomWalkerFactory(unsigned long seed, const WalkerParameters &walkerParameters, std::ostream &logger);

    /**
     * @brief Creates a new CPURandomWalker based on the parameters passed in the constructor.
     *
     * MoveGenerator and MoveFilter classes are created based on WalkerParameters::moveGeneratorParameters and
     * WalkerParameters::moveFilterParameters textual representations from the constructor.
     *
     * @return The random walker created based on the parameters from the constructor of the class
     */
    std::unique_ptr<RandomWalker> createRandomWalker() override;
};

#endif /* CPURANDOMWALKERFACTORY_H_ */
