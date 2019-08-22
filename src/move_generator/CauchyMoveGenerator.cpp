/*
 * CauchyMoveGenerator.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <cmath>

#include "CauchyMoveGenerator.h"

CauchyMoveGenerator::CauchyMoveGenerator(float width, unsigned int seed) {
    this->randomGenerator.seed(seed);
    this->cauchyDistribution = std::cauchy_distribution<float>(0.f, width);
    this->uniformAngleDistribution = std::uniform_real_distribution<float>(0.f, 2*M_PI);
}

Move CauchyMoveGenerator::generateMove() {
    float radius = this->cauchyDistribution(this->randomGenerator);
    float angle = this->uniformAngleDistribution(this->randomGenerator);

    return {radius * std::cos(angle), radius * std::sin(angle)};
}
