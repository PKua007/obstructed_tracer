/*
 * SurvivalProbability.cpp
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#include "simulation/SurvivalProbability.h"

#include <ostream>

SurvivalProbability::SurvivalProbability(double radius, std::size_t numSteps, std::size_t stepDelta,
                                         double integrationStep)

        : radius{radius}, stepDelta{stepDelta}, integrationStep{integrationStep}
{
    Expects(radius > 0);
    Expects(numSteps > 0);
    Expects(stepDelta > 0);
    Expects(integrationStep > 0);
    Expects(numSteps % stepDelta == 0);

    data.resize(numSteps / stepDelta);
}

void SurvivalProbability::store(std::ostream &out) {
    double totalTime = this->data.size() * this->stepDelta * this->integrationStep;

    for (std::size_t i; i < this->data.size(); i++) {
        double t = totalTime * (i + 1) / this->data.size();
        out << t << " " << this->data[i] << std::endl;
    }
}
