/*
 * SurvivalProbabilityAccumulator.cpp
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#include "SurvivalProbabilityAccumulator.h"
#include "utils/OMPDefines.h"

SurvivalProbabilityAccumulator::SurvivalProbabilityAccumulator(const std::vector<double> &radii, std::size_t numSteps,
                                                               std::size_t stepDelta, double integrationStep)
        : radii{radii}, numSteps{numSteps}, stepSize{stepDelta}, integrationStep{integrationStep}
{
    for (auto radius : radii)
        Expects(radius > 0);
    Expects(numSteps > 0);
    Expects(stepDelta > 0);
    Expects(integrationStep > 0);

    this->data.resize(radii.size());
    for (auto &dataEntry : this->data)
        dataEntry.resize(numSteps + 1);
}

void SurvivalProbabilityAccumulator::addTrajectories(const std::vector<Trajectory> &trajectories) {
    this->numTrajectories += trajectories.size();

    if (this->radii.empty())
        return;

    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < this->radii.size(); i++) {
        for (const auto trajectory : trajectories) {
            Assert(trajectory.getSize() >= this->numSteps + 1);
            Point start = trajectory.getFirst();

            for (std::size_t j{}; j <= this->numSteps; j++) {
                Move diff = trajectory[j * this->stepSize] - start;
                double dist2 = diff.x*diff.x + diff.y*diff.y;
                if (dist2 <= radii[i]*radii[i])
                    this->data[i][j]++;
                else
                    break;
            }
        }
    }
}

std::vector<SurvivalProbability> SurvivalProbabilityAccumulator::calculateSurvivalProbabilities() const {
    Expects(this->numTrajectories > 0);

    std::vector<SurvivalProbability> result;
    result.reserve(this->radii.size());
    for (std::size_t i{}; i < this->radii.size(); i++) {
        double radius = radii[i];
        SurvivalProbability sp(radius, this->numSteps, this->stepSize, this->integrationStep);
        for (std::size_t j{}; j < sp.size(); j++)
            sp[j] = this->data[i][j] / this->numTrajectories;
        result.push_back(sp);
    }

    return result;
}
