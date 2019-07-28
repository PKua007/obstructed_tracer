/*
 * GaussianMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef GAUSSIANMOVEGENERATOR_H_
#define GAUSSIANMOVEGENERATOR_H_

#include <random>

#include "../random_walker/MoveGenerator.h"

class GaussianMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::normal_distribution<float> normalDistribution;

public:
    GaussianMoveGenerator(float sigma, unsigned int seed);

    Move generateMove();
};

#endif /* GAUSSIANMOVEGENERATOR_H_ */
