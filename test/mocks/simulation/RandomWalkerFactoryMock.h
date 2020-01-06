/*
 * RandomWalkerFactoryMock.h
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKERFACTORYMOCK_H_
#define RANDOMWALKERFACTORYMOCK_H_

#include <catch2/trompeloeil.hpp>

#include "simulation/RandomWalkerFactory.h"

class RandomWalkerFactoryMock : public RandomWalkerFactory {
    MAKE_MOCK2(createCPURandomWalker, std::unique_ptr<RandomWalker>(unsigned long seed, const WalkerParameters &),
               override);
    MAKE_MOCK2(createGPURandomWalker, std::unique_ptr<RandomWalker>(unsigned long seed, const WalkerParameters &),
               override);
    MAKE_MOCK2(createSplitRandomWalker, std::unique_ptr<RandomWalker>(std::size_t, std::unique_ptr<RandomWalker>),
               override);
};

#endif /* RANDOMWALKERFACTORYMOCK_H_ */
