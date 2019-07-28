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

struct Point  {
    float x;
    float y;
};

std::ostream &operator<<(std::ostream &out, Point point);

using Trajectory = std::vector<Point>;

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
