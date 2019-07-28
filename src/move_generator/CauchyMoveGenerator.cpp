/*
 * CauchyMoveGenerator.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "CauchyMoveGenerator.h"

CauchyMoveGenerator::CauchyMoveGenerator(float width, unsigned int seed) {
    this->randomGenerator.seed(seed);
    this->cauchyDistribution = std::cauchy_distribution<float>(0.f, width);
}

Move CauchyMoveGenerator::generateMove() {
    return {this->cauchyDistribution(this->randomGenerator), this->cauchyDistribution(this->randomGenerator)};
}
