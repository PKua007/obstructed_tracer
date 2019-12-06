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

#include "simulation/RandomWalkerFactory.h"
#include "image/Image.h"
#include "GPURandomWalker.h"

/**
 * @brief A class which prepares GPURandomWalker.
 *
 * It has to allocate MoveGenerator and MoveFilter on GPU based on textual representations from WalkerParameters, which
 * is quite verbose process. MoveGenerator is supplied with a seed for its generator. The GPU-allocated strategies are
 * plugged into GPURandomWalker, whose rest of the parameters is determined by WalkerParamters. The class takes
 * responsibility of freeing MoveGenerator and MoveFilter memory on GPU.
 */
class GPURandomWalkerFactory : public RandomWalkerFactory {
public:
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

    public:
        MoveGeneratorOnGPUFactory(const std::string &moveGeneratorString);

        MoveGenerator *create(unsigned long seed, std::size_t numberOfWalks);
    };


    class MoveFilterOnGPUFactory {
    private:
        MoveFilterType moveFilterType{};
        BoundaryConditionsType boundaryConditionsType{};
        Image image{};

        void fetchImageData(std::istringstream &moveFilterStream, std::ostream &logger);
        void fetchBoundaryConditions(std::istringstream &moveFilterStream);

    public:
        std::size_t numberOfSetupThreads{};

        MoveFilterOnGPUFactory(const std::string &moveFilterString, std::ostream &logger);

        MoveFilter *create(unsigned long seed, std::size_t numberOfWalks);
    };


    std::mt19937 seedGenerator;
    WalkerParameters walkerParameters;
    unsigned long numberOfWalksInSeries{};
    std::ostream &logger;
    MoveGeneratorOnGPUFactory gpuMoveGeneratorFactory;
    MoveFilterOnGPUFactory gpuMoveFilterFactory;

public:
    /**
     * @brief Constructs the factory.
     *
     * It allocated @a moveGenerator and @a moveFilter on GPU (and passes seed to MoveFilter). Then it creates
     * GPURandomWalker based on @a walkerParameters and gives it @a moveGenerator and @a moveFilter. @a seed is used to
     * create byte generator, which then samples two new seeds: for MoveGenerator and MoveFilter (for
     * MoveFilter::randomValidTracer).
     *
     * @param seed the random generator seed for @a moveFilter
     * @param walkerParameters the parameters of the random walk, RandomWalker, MoveGenerator and MoveFilter
     * @param logger the output stream for some info on initializing strategies and GPURandomWalker
     */
    GPURandomWalkerFactory(unsigned long seed, const WalkerParameters &walkerParameters, std::ostream &logger);

    ~GPURandomWalkerFactory() { };

    std::unique_ptr<RandomWalker> createRandomWalker() override;
};

#endif /* GPURANDOMWALKERFACTORY_H_ */
