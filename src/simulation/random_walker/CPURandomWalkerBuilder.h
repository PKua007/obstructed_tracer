/*
 * CPURandomWalkerBuilder.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef CPURANDOMWALKERBUILDER_H_
#define CPURANDOMWALKERBUILDER_H_

#include <memory>
#include <random>
#include <iosfwd>
#include <fstream>

#include "../RandomWalkerFactory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"
#include "CPURandomWalker.h"
#include "simulation/move_generator/GaussianMoveGenerator.h"
#include "simulation/move_generator/CauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/FileUtils.h"


/**
 * @brief Traits for concrete realization of CPURandomWalkerBuilder.
 *
 * They describe concrete types of MoveFilter and MoveGenerators to instantiate: @a GaussianMoveGenerator_t,
 * @a CauchyMoveGenerator_t, @a DefaultMoveFilter_t, @a ImageMoveFilterPeriodicBC_t, @a ImageMoveFilterWallBC_t.
 * The default values can be altered by explicit specialization.
 *
 * @tparam CPURandomWalkerBuilder_t CPURandomWalkerBuilder for which we define traits
 */
template<typename CPURandomWalkerBuilder_t>
struct CPURandomWalkerBuilderTraits {
    using GaussianMoveGenerator_t = GaussianMoveGenerator;
    using CauchyMoveGenerator_t = CauchyMoveGenerator;
    using DefaultMoveFilter_t = DefaultMoveFilter;
    using ImageMoveFilterPeriodicBC_t = ImageMoveFilter<PeriodicBoundaryConditions>;
    using ImageMoveFilterWallBC_t = ImageMoveFilter<WallBoundaryConditions>;
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

    std::unique_ptr<FileIstreamProvider> fileIstreamProvider;
    std::unique_ptr<ImageReader> imageReader;
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
     * @param fileIstreamProvider the class opening file to read (for image loading)
     * @param imageReader the reader use to load the image
     * @param logger the output stream which will be passed to RandomWalker to show info
     */
    CPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                           std::unique_ptr<FileIstreamProvider> fileIstreamProvider,
                           std::unique_ptr<ImageReader> imageReader, std::ostream &logger);

    /**
     * @brief Constructs the builder with a default FileIstreamProvider and ImageReader - PPMImageReader.
     *
     * @see CPURandomWalkerBuilder(unsigned long, const RandomWalkerFactory::WalkerParameters &,
     * std::unique_ptr<FileIstreamProvider>, std::unique_ptr<ImageReader>, std::ostream &)
     */
    CPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                          std::ostream &logger)
            : CPURandomWalkerBuilder(seed, walkerParameters,
                                     std::unique_ptr<FileIstreamProvider>(new FileIstreamProvider),
                                     std::unique_ptr<ImageReader>(new PPMImageReader), logger)
    { }

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
