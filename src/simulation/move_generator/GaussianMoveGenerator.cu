/*
 * GaussianMoveGenerator.cu
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "GaussianMoveGenerator.h"

#if CUDA_HOST_COMPILATION

GaussianMoveGenerator::GaussianMoveGenerator(float sigma, float integrationStep, unsigned int seed) {
    this->randomGenerator.seed(seed);
    // We need to divide sigma by sqrt(2), because if we sample x and y with sigma^2, then r is sampled from 2sigma^2
    // And after this integraton step
    this->normalDistribution = std::normal_distribution<float>(0.f, sigma * M_SQRT1_2 * std::sqrt(integrationStep));
}

Move GaussianMoveGenerator::generateMove() {
    return {this->normalDistribution(this->randomGenerator), this->normalDistribution(this->randomGenerator)};
}

#else // CUDA_DEVICE_COMPILATION

CUDA_HOSTDEV GaussianMoveGenerator::GaussianMoveGenerator(float sigma, float integrationStep, unsigned int seed,
                                                            size_t numberOfTrajectories)
        : numberOfTrajectories{numberOfTrajectories}
{
    // Divide sigma by sqrt(2), because if we sample x and y with sigma^2, then r is sampled from 2sigma^2
    // After this, take integration step info account
    this->sigma = sigma * float{M_SQRT1_2} * sqrtf(integrationStep);
    this->states = new curandState[this->numberOfTrajectories];
    for (size_t i = 0; i < numberOfTrajectories; i++)
        curand_init(seed, i, 0, &(this->states[i]));
}

CUDA_HOSTDEV GaussianMoveGenerator::~GaussianMoveGenerator() {
    delete [] this->states;
}

CUDA_HOSTDEV Move GaussianMoveGenerator::generateMove() {
    int i = CUDA_THREAD_IDX;

    return {curand_normal(&(this->states[i])) * this->sigma, curand_normal(&(this->states[i])) * this->sigma};
}

#endif // Choice between cuda device and host compilation
