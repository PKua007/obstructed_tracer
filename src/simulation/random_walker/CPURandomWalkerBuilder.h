/*
 * CPURandomWalkerBuilder.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef CPURANDOMWALKERBUILDER_H_
#define CPURANDOMWALKERBUILDER_H_

#include <memory>
#include <iosfwd>

#include "../RandomWalkerFactory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"
#include "CPURandomWalker.h"

/**
 * @brief Traits for concrete realization of CPURandomWalkerBuilder.
 *
 * They describe concrete types of MoveFilter and MoveGenerators to instantiate: @a GaussianMoveGenerator_t,
 * @a CauchyMoveGenerator_t, @a DefaultMoveFilter_t, @a ImageMoveFilterPeriodicBC_t, @a ImageMoveFilterWallBC_t.
 *
 * @tparam CPURandomWalkerBuilder_t CPURandomWalkerBuilder for which we define traits
 */
template<typename CPURandomWalkerBuilder_t>
struct CPURandomWalkerBuilderTraits {

};

/**
 * @brief A class which construct CPURandomWalker from given parameters.
 *
 * Before creating the random walker itself, it creates CPU versions of MoveFilter and MoveGenerator based on
 * parameters and hands them to the walker.
 *
 * @tparam CPURandomWalker_t concrete CPURandomWalker to instantiate
 */
template<typename CPURandomWalker_t>
class CPURandomWalkerBuilder {
public:
    using TypeTraits = CPURandomWalkerBuilderTraits<CPURandomWalkerBuilder>;

    using GaussianMoveGenerator_t = typename TypeTraits::GaussianMoveGenerator_t;
    using CauchyMoveGenerator_t = typename TypeTraits::CauchyMoveGenerator_t;
    using DefaultMoveFilter_t = typename TypeTraits::DefaultMoveFilter_t;
    using ImageMoveFilterPeriodicBC_t = typename TypeTraits::ImageMoveFilterPeriodicBC_t;
    using ImageMoveFilterWallBC_t = typename TypeTraits::ImageMoveFilterWallBC_t;

private:
    std::mt19937 seedGenerator;
    RandomWalkerFactory::WalkerParameters walkerParameters;
    unsigned long numberOfWalksInSeries{};
    std::ostream &logger;

    std::unique_ptr<MoveGenerator> createMoveGenerator(const std::string &moveGeneratorParameters,
                                                       float integrationStep);
    std::unique_ptr<MoveFilter> createMoveFilter(const std::string &moveFilterParameters, std::ostream &logger);
    std::unique_ptr<MoveFilter> createImageMoveFilter(std::istringstream &moveFilterStream, std::ostream &logger);

public:
    /**
     * @brief Constructs the builder based on passed arguments.
     *
     * @a seed is used to create byte generator, which then will samples two new seeds: for MoveGenerator and MoveFilter
     * during creation of CPURandomWalker.
     *
     * @param seed the seed which will be used in MoveFilter and MoveGenerator
     * @param walkerParameters the parameters of the walker, MoveFilter and MoveGenerator
     * @param logger the output stream which will be passed to RandomWalker to show info
     */
    CPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                           std::ostream &logger);

    /**
     * @brief Creates a new CPURandomWalker based on the parameters passed in the constructor.
     *
     * MoveGenerator and MoveFilter classes are created based on WalkerParameters::moveGeneratorParameters and
     * WalkerParameters::moveFilterParameters textual representations from the constructor.
     *
     * @return The random walker created based on the parameters from the constructor of the class
     */
    std::unique_ptr<RandomWalker> build();
};


#include "CPURandomWalkerBuilder.tpp"

#endif /* CPURANDOMWALKERBUILDER_H_ */
