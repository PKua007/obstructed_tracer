/*
 * RandomWalkerFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKERFACTORY_H_
#define RANDOMWALKERFACTORY_H_

#include <memory>

#include "RandomWalker.h"

/**
 * @brief A class which can produce RandomWalker instances using CPU, GPU and splitting RandomWalker.
 */
class RandomWalkerFactory {
public:
    /**
     * @brief A helper struct with parameters of random walkers.
     * @see bool operator==(const RandomWalkerFactory::WalkerParameters &, const RandomWalkerFactory::WalkerParameters &)
     * @see std::ostream &operator<<(std::ostream &, const RandomWalkerFactory::WalkerParameters &)
     */
    struct WalkerParameters {
        /**
         * @brief The textual representation of generator parameters.
         */
        std::string moveGeneratorParameters;

        /**
         * @brief The textual representation of move filter parameters.
         */
        std::string moveFilterParameters;

        /**
         * @brief The number of walks to be performed in a single series of walks.
         */
        std::size_t numberOfWalksInSeries;

        /**
         * @brief The parameters of the walks.
         */
        RandomWalker::WalkParameters walkParameters;
    };

    virtual ~RandomWalkerFactory() = default;

    /**
     * @brief Creates new RandomWalker using CPU corresponding to concrete factory.
     *
     * @param seed random seed to initialize RandomWalker
     * @param walkerParameters the parameters of the walker
     */
    virtual std::unique_ptr<RandomWalker> createCPURandomWalker(unsigned long seed,
                                                                const WalkerParameters &walkerParameters) = 0;

    /**
     * @brief Creates new RandomWalker using GPU  corresponding to concrete factory.
     *
     * @param seed random seed to initialize RandomWalker
     * @param walkerParameters the parameters of the walker
     */
    virtual std::unique_ptr<RandomWalker> createGPURandomWalker(unsigned long seed,
                                                                const WalkerParameters &walkerParameters) = 0;

    /**
     * @brief Creates new RandomWalker which uses @a randomWalker to simulate whole trajectories in a few splits
     * corresponding to concrete factory.
     *
     * Each part of some number of trajectories is simulated in one run of @a randomWalker.
     *
     * @param numberOfSplits number of parts to divide trajectories into
     * @param randomWalker underlying RandomWalker to use
     *
     * @see SplitRandomWalker
     */
    virtual std::unique_ptr<RandomWalker> createSplitRandomWalker(std::size_t numberOfSplits,
                                                                  std::unique_ptr<RandomWalker> randomWalker) = 0;
};

/**
 * @brief Compares two RandomWalkerFactory::WalkerParameters.
 */
inline bool operator==(const RandomWalkerFactory::WalkerParameters &wp1,
                       const RandomWalkerFactory::WalkerParameters &wp2)
{
    return wp1.moveGeneratorParameters == wp2.moveGeneratorParameters
           && wp1.moveFilterParameters == wp2.moveFilterParameters
           && wp1.numberOfWalksInSeries == wp2.numberOfWalksInSeries
           && wp1.walkParameters == wp2.walkParameters;
}

/**
 * @brief Stream insertion operator printing walker parameters.
 * @param out stream to print to
 * @param walkerParameters walker parameters to be printed
 * @return the reference to @a out
 */
inline std::ostream &operator<<(std::ostream &out, const RandomWalkerFactory::WalkerParameters &walkerParameters) {
    out << "{moveGeneratorParameters: " << walkerParameters.moveGeneratorParameters
        << ", moveFilterParameters: " << walkerParameters.moveFilterParameters
        << ", numberOfWalksInSeries: " << walkerParameters.numberOfWalksInSeries
        << ", walkParameters: " << walkerParameters.walkParameters << "}";
    return out;
}

#endif /* RANDOMWALKERFACTORY_H_ */
