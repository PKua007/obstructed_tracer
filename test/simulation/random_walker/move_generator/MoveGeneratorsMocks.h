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
#include "../../../test_utils/GPURTTI.h"

struct GaussianMoveGeneratorMock : public MoveGenerator, GPUNamedClass {
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
    CUDA_IMPLEMENT_GET_CLASS_NAME("GaussianMoveGeneratorMock");
};

struct CauchyMoveGeneratorMock : public MoveGenerator, GPUNamedClass {
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
    CUDA_IMPLEMENT_GET_CLASS_NAME("CauchyMoveGeneratoMock");
};

#endif /* MOVEGENERATORSMOCKS_H_ */
