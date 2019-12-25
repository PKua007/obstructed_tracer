/*
 * GPURandomWalkerBuilder.cu
 *
 *  Created on: 23 gru 2019
 *      Author: pkua
 */

#include "GPURandomWalkerBuilder.h"

#include "simulation/move_generator/GaussianMoveGenerator.h"
#include "simulation/move_generator/CauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"

#include "GPURandomWalkerBuilder.tpp"
#include "GPURandomWalker.h"

template class GPURandomWalkerBuilder<GPURandomWalker>;
