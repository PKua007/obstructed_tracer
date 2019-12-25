/*
 * MoveGeneratorsMocks.h
 *
 *  Created on: 25 gru 2019
 *      Author: pkua
 */

#ifndef MOVEGENERATORSMOCKS_H_
#define MOVEGENERATORSMOCKS_H_

#include "simulation/MoveGenerator.h"
#include "utils/CudaDefines.h"

struct CPUGaussianMoveGeneratorMock : public MoveGenerator {
    float sigma;
    float integrationStep;
    unsigned int seed;

    CUDA_HOSTDEV CPUGaussianMoveGeneratorMock(float sigma, float integrationStep, unsigned int seed)
            : sigma{sigma}, integrationStep{integrationStep}, seed{seed}
    { }

    CUDA_HOSTDEV Move generateMove() override { return Move{}; }
};

struct CPUCauchyMoveGeneratorMock : public MoveGenerator {
    float width;
    float integrationStep;
    unsigned int seed;

    CUDA_HOSTDEV CPUCauchyMoveGeneratorMock(float width, float integrationStep, unsigned int seed)
            : width{width}, integrationStep{integrationStep}, seed{seed}
    { }

    CUDA_HOSTDEV Move generateMove() override { return Move{}; }
};

#endif /* MOVEGENERATORSMOCKS_H_ */
