/*
 * RandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>

#include "RandomWalker.h"

float RandomWalker::nextGaussian() {
    return this->normalDistribution(this->randomGenerator);
}

RandomWalker::RandomWalker(float initX, float initY, float distributionVariance, std::size_t numberOfSteps) :
        initX{initX}, initY{initY}, numberOfSteps{numberOfSteps} {
    std::random_device randomDevice;
    this->randomGenerator.seed(randomDevice());
    this->normalDistribution = std::normal_distribution<float>(0.f, distributionVariance);
}

Trajectory RandomWalker::run() {
    Trajectory trajectory;
    trajectory.reserve(this->numberOfSteps);

    Point tracer = {this->initX, this->initY};
    trajectory.push_back(tracer);
    for (std::size_t i = 0; i < this->numberOfSteps; i++) {
        float deltaX = this->nextGaussian();
        float deltaY = this->nextGaussian();
        tracer.x += deltaX;
        tracer.y += deltaY;
        trajectory.push_back(tracer);
    }

    return trajectory;
}

std::ostream &operator<<(std::ostream& out, Point point) {
    out << "{" << point.x << ", " << point.y << "}";
    return out;
}
