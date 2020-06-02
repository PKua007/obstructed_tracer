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
        dataEntry.resize(numSteps + 1);     // +1 for t = 0
}

void SurvivalProbabilityAccumulator::addTrajectories(const std::vector<Trajectory> &trajectories) {
    this->numTrajectories += trajectories.size();

    if (this->radii.empty())
        return;

    _OMP_PARALLEL_FOR
    for (std::size_t radiusIdx = 0; radiusIdx < this->radii.size(); radiusIdx++) {
        for (const auto trajectory : trajectories) {
            Assert(trajectory.getSize() >= this->numSteps + 1);
            Point start = trajectory.getFirst();

            // We go from 0 to this->numSteps inclusive, because index 0 corresponds to t=0 and last to t=tmax
            for (std::size_t step{}; step <= this->numSteps; step++) {
                Move diff = trajectory[step * this->stepSize] - start;
                double displacement2 = diff.x*diff.x + diff.y*diff.y;
                if (displacement2 <= radii[radiusIdx]*radii[radiusIdx])
                    this->data[radiusIdx][step]++;
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
    for (std::size_t radiusIdx{}; radiusIdx < this->radii.size(); radiusIdx++) {
        double radius = radii[radiusIdx];
        SurvivalProbability sp(radius, this->numSteps, this->stepSize, this->integrationStep);
        for (std::size_t j{}; j < sp.size(); j++)
            sp[j] = this->data[radiusIdx][j] / this->numTrajectories;
        result.push_back(sp);
    }

    return result;
}
