/*
 * RandomWalkerFactoryImpll.cpp
 *
 *  Created on: 23 gru 2019
 *      Author: pkua
 */

#include "RandomWalkerFactoryImpl.h"
#include "GPURandomWalkerBuilder.h"
#include "SplitRandomWalker.h"
#include "simulation/move_generator/gpu/GPUGaussianMoveGenerator.h"
#include "simulation/move_generator/gpu/GPUCauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"

/* Concrete type traits for GPURandomWalkerBuilder with "production" classes */
template<>
struct GPURandomWalkerBuilderTraits<GPURandomWalkerBuilder<GPURandomWalker>> {
    using GaussianMoveGenerator_t = GPUGaussianMoveGenerator;
    using CauchyMoveGenerator_t = GPUCauchyMoveGenerator;
    using DefaultMoveFilter_t = DefaultMoveFilter;
    using ImageMoveFilterPeriodicBC_t = ImageMoveFilter<PeriodicBoundaryConditions>;
    using ImageMoveFilterWallBC_t = ImageMoveFilter<WallBoundaryConditions>;
};


std::unique_ptr<RandomWalker> RandomWalkerFactoryImpl::createGPURandomWalker(unsigned long seed,
                                                                             const WalkerParameters &walkerParameters)
{
    return GPURandomWalkerBuilder<GPURandomWalker>(seed, walkerParameters, this->logger).build();
}
