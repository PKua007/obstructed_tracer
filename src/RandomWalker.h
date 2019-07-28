/*
 * RandomWalker.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKER_H_
#define RANDOMWALKER_H_

#include <vector>
#include <array>
#include <random>

#include "Trajectory.h"

class RandomWalker {
private:
    float initX;
    float initY;
    std::size_t numberOfSteps;

    std::mt19937 randomGenerator;
    std::normal_distribution<float> normalDistribution;

    float nextGaussian();

public:
    RandomWalker(float initX, float initY, float standardDeviation, std::size_t numberOfSteps);

    Trajectory run();
};

#endif /* RANDOMWALKER_H_ */
