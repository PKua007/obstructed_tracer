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
#include "../../../utils/GPUMock.h"

struct GaussianMoveGeneratorMock : public MoveGenerator, GPUMock {
    float sigma;
    float integrationStep;
    unsigned int seed;
    size_t numberOfTrajectories;

    CUDA_HOST GaussianMoveGeneratorMock(float sigma, float integrationStep, unsigned int seed)
            : sigma{sigma}, integrationStep{integrationStep}, seed{seed}
    { }

    CUDA_DEV GaussianMoveGeneratorMock(float sigma, float integrationStep, unsigned int seed,
                                       size_t numberOfTrajectories)
            : sigma{sigma}, integrationStep{integrationStep}, seed{seed}, numberOfTrajectories{numberOfTrajectories}
    { }

    CUDA_HOSTDEV Move generateMove() override { return Move{}; }
    CUDA_DEV char *getClassName(char *nameOut, size_t maxSize) const override {
        return this->getClassName0("GaussianMoveGeneratorMock", nameOut, maxSize);
    }
};

struct CauchyMoveGeneratorMock : public MoveGenerator, GPUMock {
    float width;
    float integrationStep;
    unsigned int seed;
    size_t numberOfTrajectories;

    CUDA_HOST CauchyMoveGeneratorMock(float width, float integrationStep, unsigned int seed)
            : width{width}, integrationStep{integrationStep}, seed{seed}
    { }

    CUDA_DEV CauchyMoveGeneratorMock(float width, float integrationStep, unsigned int seed,
                                     size_t numberOfTrajectories)
            : width{width}, integrationStep{integrationStep}, seed{seed}, numberOfTrajectories{numberOfTrajectories}
    { }

    CUDA_HOSTDEV Move generateMove() override { return Move{}; }
    CUDA_DEV char *getClassName(char *nameOut, size_t maxSize) const override {
        return this->getClassName0("CauchyMoveGeneratorMock", nameOut, maxSize);
    }
};

#endif /* MOVEGENERATORSMOCKS_H_ */
