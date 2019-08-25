/*
 * GaussianMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef GAUSSIANMOVEGENERATOR_H_
#define GAUSSIANMOVEGENERATOR_H_

#include <random>

#include "simulation/MoveGenerator.h"

class GaussianMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::normal_distribution<float> normalDistribution;

public:
    GaussianMoveGenerator(float sigma, unsigned int seed);

    Move generateMove() override;
};

#endif /* GAUSSIANMOVEGENERATOR_H_ */
