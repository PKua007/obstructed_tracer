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
#include "CPURandomWalkerBuilder.h"
#include "CPURandomWalker.h"
#include "GPURandomWalkerBuilder.h"
#include "GPURandomWalker.h"
#include "SplitRandomWalker.h"

extern template class CPURandomWalkerBuilder<CPURandomWalker>;
extern template class GPURandomWalkerBuilder<CPURandomWalker>;

class RandomWalkerFactoryImpl : public RandomWalkerFactory {
private:
    std::ostream &logger;

public:
    explicit RandomWalkerFactoryImpl(std::ostream &logger) : logger{logger} { };
    virtual ~RandomWalkerFactoryImpl() { };

    std::unique_ptr<RandomWalker> createCPURandomWalker(unsigned long seed,
                                                        const WalkerParameters &walkerParameters) override
    {
        return CPURandomWalkerBuilder<CPURandomWalker>(seed, walkerParameters, this->logger).build();
    }

    std::unique_ptr<RandomWalker> createGPURandomWalker(unsigned long seed,
                                                        const WalkerParameters &walkerParameters) override
    {
        return GPURandomWalkerBuilder<GPURandomWalker>(seed, walkerParameters, this->logger).build();
    }

    std::unique_ptr<RandomWalker> createSplitRandomWalker(std::size_t numberOfSplits,
                                                          std::unique_ptr<RandomWalker> randomWalker) override
    {
        return std::unique_ptr<RandomWalker>(new SplitRandomWalker(numberOfSplits, std::move(randomWalker)));
    }

};

#endif /* RANDOMWALKERFACTORYIMPL_H_ */
