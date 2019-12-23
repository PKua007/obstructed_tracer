/*
 * RandomWalkerFactoryImpl.h
 *
 *  Created on: 20 gru 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKERFACTORYIMPL_H_
#define RANDOMWALKERFACTORYIMPL_H_

#include <iosfwd>

#include "../RandomWalkerFactory.h"

class RandomWalkerFactoryImpl : public RandomWalkerFactory {
private:
    std::ostream &logger;

public:
    explicit RandomWalkerFactoryImpl(std::ostream &logger) : logger{logger} { };
    virtual ~RandomWalkerFactoryImpl() { };

    std::unique_ptr<RandomWalker> createCPURandomWalker(unsigned long seed,
                                                        const WalkerParameters &walkerParameters) override;
    std::unique_ptr<RandomWalker> createGPURandomWalker(unsigned long seed,
                                                        const WalkerParameters &walkerParameters) override;
    std::unique_ptr<RandomWalker> createSplitRandomWalker(std::size_t numberOfSplits,
                                                          std::unique_ptr<RandomWalker> randomWalker) override;
};

#endif /* RANDOMWALKERFACTORYIMPL_H_ */
