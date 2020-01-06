/*
 * GPURandomWalkerBuilder.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPURANDOMWALKERBUILDER_H_
#define GPURANDOMWALKERBUILDER_H_

#include <memory>
#include <random>

#include "../MoveGenerator.h"
#include "../MoveFilter.h"
#include "../RandomWalkerFactory.h"
#include "simulation/move_generator/GaussianMoveGenerator.h"
#include "simulation/move_generator/CauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/Image.h"
#include "image/PPMImageReader.h"
#include "utils/FileUtils.h"


/**
 * @brief Traits for concrete realization of GPURandomWalkerBuilder.
 *
 * They describe concrete types of MoveFilter and MoveGenerators to instantiate: @a GaussianMoveGenerator_t,
 * @a CauchyMoveGenerator_t, @a DefaultMoveFilter_t, @a ImageMoveFilterPeriodicBC_t, @a ImageMoveFilterWallBC_t.
 * The default values can be altered by explicit specialization.
 *
 * @tparam GPURandomWalkerBuilder_t GPURandomWalkerBuilder for which we define traits
 */
template<typename GPURandomWalkerBuilder_t>
struct GPURandomWalkerBuilderTraits {
    using GaussianMoveGenerator_t = GaussianMoveGenerator;
    using CauchyMoveGenerator_t = CauchyMoveGenerator;
    using DefaultMoveFilter_t = DefaultMoveFilter;
    using ImageMoveFilterPeriodicBC_t = ImageMoveFilter<PeriodicBoundaryConditions>;
    using ImageMoveFilterWallBC_t = ImageMoveFilter<WallBoundaryConditions>;
};

/**
 * @brief A class which prepares GPURandomWalker.
 *
 * It has to allocate MoveGenerator and MoveFilter on GPU based on textual representations from
 * RandomWalkerFactory::WalkerParameters, which is quite a verbose process. MoveGenerator is supplied with a seed for
 * its generator. The GPU-allocated strategies are plugged into GPURandomWalker, whose rest of the parameters is
 * determined by WalkerParamters.
 *
 * @tparam GPURandomWalker_t concrete GPURandomWalker to instantiate
 */
template<typename GPURandomWalker_t>
class GPURandomWalkerBuilder {
public:
    using TypeTraits = GPURandomWalkerBuilderTraits<GPURandomWalkerBuilder>;

    using GaussianMoveGenerator_t = typename TypeTraits::GaussianMoveGenerator_t;
    using CauchyMoveGenerator_t = typename TypeTraits::CauchyMoveGenerator_t;
    using DefaultMoveFilter_t = typename TypeTraits::DefaultMoveFilter_t;
    using ImageMoveFilterPeriodicBC_t = typename TypeTraits::ImageMoveFilterPeriodicBC_t;
    using ImageMoveFilterWallBC_t = typename TypeTraits::ImageMoveFilterWallBC_t;

    enum MoveGeneratorType {
        GAUSSIAN,
        CAUCHY
    };

    enum MoveFilterType {
        DEFAULT,
        IMAGE
    };

    enum BoundaryConditionsType {
        WALL,
        PERIODIC
    };

private:
    class MoveGeneratorOnGPUFactory {
    private:
        MoveGeneratorType moveGeneratorType{};
        float sigma{};
        float integrationStep{};

    public:
        MoveGeneratorOnGPUFactory(const std::string &moveGeneratorString, float integrationStep);

        MoveGenerator *create(unsigned long seed, std::size_t numberOfWalks);
    };


    class MoveFilterOnGPUFactory {
    private:
        MoveFilterType moveFilterType{};
        BoundaryConditionsType boundaryConditionsType{};
        Image image{};

        std::unique_ptr<FileIstreamProvider> fileIstreamProvider;
        std::unique_ptr<ImageReader> imageReader;

        void fetchImageData(std::istringstream &moveFilterStream, std::ostream &logger);
        void fetchBoundaryConditions(std::istringstream &moveFilterStream);

    public:
        std::size_t numberOfSetupThreads{};

        MoveFilterOnGPUFactory(const std::string &moveFilterString, std::ostream &logger,
                               std::unique_ptr<FileIstreamProvider> fileIstreamProvider,
                               std::unique_ptr<ImageReader> imageReader);

        MoveFilter *create(unsigned long seed, std::size_t numberOfWalks);
    };


    std::mt19937 seedGenerator;
    RandomWalkerFactory::WalkerParameters walkerParameters;
    unsigned long numberOfWalksInSeries{};
    std::ostream &logger;

    MoveGeneratorOnGPUFactory gpuMoveGeneratorFactory;
    MoveFilterOnGPUFactory gpuMoveFilterFactory;

public:
    /**
     * @brief Constructs the builder.
     *
     * @a seed is used to create byte generator, which then will be used to sample two new seeds: for MoveGenerator and
     * MoveFilter (for MoveFilter::randomValidTracer).
     *
     * @param seed the random generator seed for @a moveFilter
     * @param walkerParameters the parameters of the random walk, RandomWalker, MoveGenerator and MoveFilter
     * @param walkerParameters the parameters of the walker, MoveFilter and MoveGenerator
     * @param fileIstreamProvider the class opening file to read (for image loading)
     * @param imageReader the reader use to load the image
     * @param logger the output stream for some info on initializing strategies and GPURandomWalker
     */
    GPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                           std::unique_ptr<FileIstreamProvider> fileIstreamProvider,
                           std::unique_ptr<ImageReader> imageReader, std::ostream &logger);

    /**
     * @brief Constructs the builder with a default FileIstreamProvider and ImageReader - PPMImageReader.
     *
     * @see GPURandomWalkerBuilder(unsigned long, const RandomWalkerFactory::WalkerParameters &,
     * std::unique_ptr<FileIstreamProvider>, std::unique_ptr<ImageReader>, std::ostream &)
     */
    GPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                          std::ostream &logger)
            : GPURandomWalkerBuilder(seed, walkerParameters,
                                     std::unique_ptr<FileIstreamProvider>(new FileIstreamProvider),
                                     std::unique_ptr<ImageReader>(new PPMImageReader), logger)
    { }

    ~GPURandomWalkerBuilder() { };

    /**
     * @brief Creates GPURandomWalker based on the paramters passed in the constructor of the class.
     *
     * It allocated MoveGenerator and MoveFilter on GPU (and passes randomly sampled seeds to them). Then it creates
     * RandomWalker based on RandomWalkerFactory::WalkerParameters and gives it MoveGenerator and MoveFilter. The class
     * passes the responsibility of freeing MoveGenerator and MoveFilter memory on GPU to GPURandomWalker.
     *
     * @return GPURandomWalker based on the parameters passed in the constructor of the class
     */
    std::unique_ptr<RandomWalker> build();
};

#include "GPURandomWalkerBuilder.tpp"

#endif /* GPURANDOMWALKERBUILDER_H_ */
