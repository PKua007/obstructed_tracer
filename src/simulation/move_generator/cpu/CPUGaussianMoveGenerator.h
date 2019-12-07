/*
 * CPUGaussianMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef CPUGAUSSIANMOVEGENERATOR_H_
#define CPUGAUSSIANMOVEGENERATOR_H_

#include <random>

#include "simulation/MoveGenerator.h"

/**
 * @brief Generates moves on CPU according to normal distribution in radius and uniform distribution in angle.
 */
class CPUGaussianMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::normal_distribution<float> normalDistribution;

public:
    /**
     * @brief Initializes the generators using seeds generated by byte generator seeded with @a seed parameter
     *
     * @param sigma the standard deviation in normal distribution
     * @param integrationStep the integration step in the diffusion used to rescale the distribution properly - in this
     * case by square root of the integration step
     * @param seed the random seed for generators
     */
    CPUGaussianMoveGenerator(float sigma, float integrationStep, unsigned int seed);

    /**
     * @brief Generates random move on CPU according to normal distribution in radius and uniform distribution in angle.
     *
     * @return random move according to normal distribution in radius and uniform distribution in angle
     */
    Move generateMove() override;
};

#endif /* CPUGAUSSIANMOVEGENERATOR_H_ */
