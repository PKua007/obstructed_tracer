/*
 * SurvivalProbability.cpp
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#include "simulation/SurvivalProbability.h"

#include <ostream>

SurvivalProbability::SurvivalProbability(double radius, std::size_t numSteps, std::size_t stepSize,
                                         double integrationStep)

        : radius{radius}, stepSize{stepSize}, integrationStep{integrationStep}
{
    Expects(radius > 0);
    Expects(numSteps > 0);
    Expects(stepSize > 0);
    Expects(integrationStep > 0);

    data.resize(numSteps + 1);
}

void SurvivalProbability::store(std::ostream &out) const {
    for (std::size_t i; i < this->data.size(); i++) {
        double t = i * this->stepSize * this->integrationStep;
        out << t << " " << this->data[i] << std::endl;
    }
}
