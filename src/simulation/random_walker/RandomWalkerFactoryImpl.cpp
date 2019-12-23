/*
 * RandomWalkerFactoryImpl.cu
 *
 *  Created on: 23 gru 2019
 *      Author: pkua
 */

#include "RandomWalkerFactoryImpl.h"
#include "CPURandomWalkerBuilder.h"
#include "GPURandomWalkerBuilder.h"
#include "SplitRandomWalker.h"
#include "simulation/move_generator/cpu/CPUGaussianMoveGenerator.h"
#include "simulation/move_generator/cpu/CPUCauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"

/* Concrete type traits for CPURandomWalkerBuilder with "production" classes */
template<>
struct CPURandomWalkerBuilderTraits<CPURandomWalkerBuilder<CPURandomWalker>> {
    using GaussianMoveGenerator_t = CPUGaussianMoveGenerator;
    using CauchyMoveGenerator_t = CPUCauchyMoveGenerator;
    using DefaultMoveFilter_t = DefaultMoveFilter;
    using ImageMoveFilterPeriodicBC_t = ImageMoveFilter<PeriodicBoundaryConditions>;
    using ImageMoveFilterWallBC_t = ImageMoveFilter<WallBoundaryConditions>;
};

std::unique_ptr<RandomWalker> RandomWalkerFactoryImpl::createCPURandomWalker(unsigned long seed,
                                                                             const WalkerParameters &walkerParameters)
{
    return CPURandomWalkerBuilder<CPURandomWalker>(seed, walkerParameters, this->logger).build();
}

std::unique_ptr<RandomWalker> RandomWalkerFactoryImpl::createGPURandomWalker(unsigned long seed,
                                                                             const WalkerParameters &walkerParameters)
{
    return GPURandomWalkerBuilder(seed, walkerParameters, this->logger).build();
}

std::unique_ptr<RandomWalker>
RandomWalkerFactoryImpl::createSplitRandomWalker(std::size_t numberOfSplits, std::unique_ptr<RandomWalker> randomWalker)
{
    return std::unique_ptr<RandomWalker>(new SplitRandomWalker(numberOfSplits, std::move(randomWalker)));
}
