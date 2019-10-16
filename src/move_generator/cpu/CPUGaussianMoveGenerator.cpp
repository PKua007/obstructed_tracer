/*
 * CPUGaussianMoveGenerator.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "CPUGaussianMoveGenerator.h"

CPUGaussianMoveGenerator::CPUGaussianMoveGenerator(float sigma, unsigned int seed) {
    this->randomGenerator.seed(seed);
    // We need to divide sigma by sqrt(2), because if we sample x and y with sigma^2, then r is sampled from 2sigma^2
    this->normalDistribution = std::normal_distribution<float>(0.f, sigma * M_SQRT1_2);
}

Move CPUGaussianMoveGenerator::generateMove() {
    return {this->normalDistribution(this->randomGenerator), this->normalDistribution(this->randomGenerator)};
}
