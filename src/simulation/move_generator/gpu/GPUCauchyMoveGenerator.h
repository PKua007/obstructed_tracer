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

/**
 * @brief A `__device__` class generating moves on GPU according to Cauchy distribution in radius and uniform
 * distribution in angle.
 */
class GPUCauchyMoveGenerator : public MoveGenerator {
private:
    float width;
    curandState *states;
    size_t numberOfTrajectories;

    CUDA_DEV float randomCauchy();

public:
    /**
     * @brief Initializes the generators using seeds generated by byte generator seeded with @a seed parameter.
     *
     * The constructor is expected to be called only by the first CUDA thread.
     *
     * @param width the width parameter in Cauchy distribution
     * @param integrationStep the integration step in the diffusion used to rescale the distribution properly - in this
     * case by square root of the integration step
     * @param seed the random seed for generators
     * @param numberOfTrajectories the number of trajectories for which independent number will be sampled on GPU
     */
    CUDA_DEV GPUCauchyMoveGenerator(float width, float integrationStep, unsigned int seed, size_t numberOfTrajectories);

    CUDA_DEV GPUCauchyMoveGenerator(const GPUCauchyMoveGenerator &other) = delete;
    CUDA_DEV GPUCauchyMoveGenerator operator=(GPUCauchyMoveGenerator other) = delete;

    /**
     * @brief The destructor which is expected to be called only in the first CUDA thread.
     */
    CUDA_DEV ~GPUCauchyMoveGenerator();

    /**
     * @brief Generates random move on CPU according to Cauchy distribution in radius and uniform distribution in angle.
     *
     * It respects the id of CUDA thread and for each thread the sequences are independent.
     *
     * @return random move according to Cauchy distribution in radius and uniform distribution in angle
     */
    CUDA_DEV Move generateMove() override;
};

#endif /* GPUCAUCHYMOVEGENERATOR_H_ */
