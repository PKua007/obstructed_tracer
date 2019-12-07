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
 * It has to allocate MoveGenerator and MoveFilter on GPU based on textual representations from
 * RandomWalkerFactory::WalkerParameters, which is quite a verbose process. MoveGenerator is supplied with a seed for
 * its generator. The GPU-allocated strategies are plugged into GPURandomWalker, whose rest of the parameters is
 * determined by WalkerParamters.
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
     * @a seed is used to create byte generator, which then will be used to sample two new seeds: for MoveGenerator and
     * MoveFilter (for MoveFilter::randomValidTracer).
     *
     * @param seed the random generator seed for @a moveFilter
     * @param walkerParameters the parameters of the random walk, RandomWalker, MoveGenerator and MoveFilter
     * @param logger the output stream for some info on initializing strategies and GPURandomWalker
     */
    GPURandomWalkerFactory(unsigned long seed, const WalkerParameters &walkerParameters, std::ostream &logger);

    ~GPURandomWalkerFactory() { };

    /**
     * @brief Creates GPURandomWalker based on the paramters passed in the constructor of the class.
     *
     * It allocated MoveGenerator and MoveFilter on GPU (and passes randomly sampled seeds to them). Then it creates
     * RandomWalker based on RandomWalkerFactory::WalkerParameters and gives it MoveGenerator and MoveFilter. The class
     * passes the responsibility of freeing MoveGenerator and MoveFilter memory on GPU to GPURandomWalker.
     *
     * @return GPURandomWalker based on the parameters passed in the constructor of the class
     */
    std::unique_ptr<RandomWalker> createRandomWalker() override;
};

#endif /* GPURANDOMWALKERFACTORY_H_ */
