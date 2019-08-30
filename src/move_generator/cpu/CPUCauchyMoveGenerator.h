/*
 * CPUCauchyMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef CPUCAUCHYMOVEGENERATOR_H_
#define CPUCAUCHYMOVEGENERATOR_H_

#include <random>
#include <cmath>

#include "simulation/MoveGenerator.h"

class CPUCauchyMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::cauchy_distribution<float> cauchyDistribution;
    std::uniform_real_distribution<float> uniformAngleDistribution;

public:
    CPUCauchyMoveGenerator(float width, unsigned int seed);

    Move generateMove() override;
};

#endif /* CPUCAUCHYMOVEGENERATOR_H_ */
