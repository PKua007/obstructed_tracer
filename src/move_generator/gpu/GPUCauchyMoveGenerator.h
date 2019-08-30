/*
 * GPUCauchyMoveGenerator.h
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#ifndef GPUCAUCHYMOVEGENERATOR_H_
#define GPUCAUCHYMOVEGENERATOR_H_


#include <curand_kernel.h>

#include "simulation/MoveGenerator.h"


class GPUCauchyMoveGenerator : public MoveGenerator {
private:
    float sigma;
    curandState *states;
    size_t numberOfTrajectories;

    CUDA_DEV float randomCauchy();

public:
    CUDA_DEV GPUCauchyMoveGenerator(float sigma, unsigned int seed, size_t numberOfTrajectories);
    CUDA_DEV GPUCauchyMoveGenerator(const GPUCauchyMoveGenerator &other) = delete;
    CUDA_DEV GPUCauchyMoveGenerator operator=(GPUCauchyMoveGenerator other) = delete;
    CUDA_DEV ~GPUCauchyMoveGenerator();

    CUDA_DEV Move generateMove() override;
};

#endif /* GPUCAUCHYMOVEGENERATOR_H_ */
