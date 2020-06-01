/*
 * SurvivalProbabilityAccumulator.cpp
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#include <simulation/SurvivalProbabilityAccumulator.h>

SurvivalProbabilityAccumulator::SurvivalProbabilityAccumulator(const std::vector<double> &radii, std::size_t numSteps,
                                                               std::size_t stepDelta, double integrationStep)
{

}

void SurvivalProbabilityAccumulator::addTrajectories(const std::vector<Trajectory> &trajectories) {
}

std::vector<SurvivalProbability> SurvivalProbabilityAccumulator::calculateSurvivalProbabilities() const {
    return {};
}
