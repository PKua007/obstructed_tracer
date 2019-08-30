/*
 * CPUGaussianMoveGenerator.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "CPUGaussianMoveGenerator.h"

CPUGaussianMoveGenerator::CPUGaussianMoveGenerator(float sigma, unsigned int seed) {
    this->randomGenerator.seed(seed);
    this->normalDistribution = std::normal_distribution<float>(0.f, sigma);
}

Move CPUGaussianMoveGenerator::generateMove() {
    return {this->normalDistribution(this->randomGenerator), this->normalDistribution(this->randomGenerator)};
}
