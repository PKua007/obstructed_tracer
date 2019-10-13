/*
 * RandomWalkerFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKERFACTORY_H_
#define RANDOMWALKERFACTORY_H_

#include "RandomWalker.h"

/**
 * @brief A class which can produce RandomWalker instances of a specific kind.
 */
class RandomWalkerFactory {
public:
    /**
     * @brief A helper struct with parameters of random walkers.
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
     * @brief Gets the random walker produced by the factory.
     */
    virtual RandomWalker &getRandomWalker() = 0;
};

#endif /* RANDOMWALKERFACTORY_H_ */
