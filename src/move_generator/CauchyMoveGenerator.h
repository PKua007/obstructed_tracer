/*
 * CauchyMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef CAUCHYMOVEGENERATOR_H_
#define CAUCHYMOVEGENERATOR_H_

#include <random>
#include <cmath>

#include "../simulation/MoveGenerator.h"

class CauchyMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::cauchy_distribution<float> cauchyDistribution;
    std::uniform_real_distribution<float> uniformAngleDistribution;

public:
    CauchyMoveGenerator(float width, unsigned int seed);

    Move generateMove() override;
};

#endif /* CAUCHYMOVEGENERATOR_H_ */
