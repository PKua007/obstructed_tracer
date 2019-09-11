/*
 * CPUGaussianMoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef CPUGAUSSIANMOVEGENERATOR_H_
#define CPUGAUSSIANMOVEGENERATOR_H_

#include <random>

#include "random_walker/MoveGenerator.h"

class CPUGaussianMoveGenerator : public MoveGenerator {
private:
    std::mt19937 randomGenerator;
    std::normal_distribution<float> normalDistribution;

public:
    CPUGaussianMoveGenerator(float sigma, unsigned int seed);

    Move generateMove() override;
};

#endif /* CPUGAUSSIANMOVEGENERATOR_H_ */
