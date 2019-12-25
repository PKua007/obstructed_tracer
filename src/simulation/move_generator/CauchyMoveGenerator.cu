/*
 * CauchyMoveGenerator.cu
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <cmath>
#include <math_constants.h>

#include "CauchyMoveGenerator.h"

#if CUDA_HOST_COMPILATION

CauchyMoveGenerator::CauchyMoveGenerator(float width, float integrationStep, unsigned int seed) {
    this->randomGenerator.seed(seed);
    this->cauchyDistribution = std::cauchy_distribution<float>(0.f, width * integrationStep);
    this->uniformAngleDistribution = std::uniform_real_distribution<float>(0.f, 2*M_PI);
}

Move CauchyMoveGenerator::generateMove() {
    float radius = this->cauchyDistribution(this->randomGenerator);
    float angle = this->uniformAngleDistribution(this->randomGenerator);

    return {radius * std::cos(angle), radius * std::sin(angle)};
}

#else // CUDA_DEVICE_COMPILATION

CUDA_HOSTDEV CauchyMoveGenerator::CauchyMoveGenerator(float width, float integrationStep, unsigned int seed,
                                                        size_t numberOfTrajectories)
        : width{width * integrationStep}, numberOfTrajectories{numberOfTrajectories}
{
    this->states = new curandState[this->numberOfTrajectories];
    for (size_t i = 0; i < numberOfTrajectories; i++)
        curand_init(seed, i, 0, &(this->states[i]));
}

CUDA_HOSTDEV CauchyMoveGenerator::~CauchyMoveGenerator() {
    delete [] this->states;
}

CUDA_HOSTDEV float CauchyMoveGenerator::randomCauchy() {
    float uniform = curand_uniform(&(this->states[CUDA_THREAD_IDX]));
    return this->width * tanf(CUDART_PI_F * (uniform - 0.5f));
}

CUDA_HOSTDEV Move CauchyMoveGenerator::generateMove() {
    float radius = this->randomCauchy();
    float angle = 2 * CUDART_PI_F * curand_uniform(&(this->states[CUDA_THREAD_IDX]));

    return {radius * cosf(angle), radius * sinf(angle)};
}

#endif // Choice between cuda device and host compilation
