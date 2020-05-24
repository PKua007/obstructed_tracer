/*
 * TimeAveragedMSDCalculator.cpp
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#include "TimeAveragedMSDCalculator.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"

TimeAveragedMSDCalculator::TimeAveragedMSDCalculator(std::size_t maxDelta, std::size_t deltaStep, float integrationStep)
        : maxDelta{maxDelta}, deltaStep{deltaStep}, integrationStep{integrationStep}
{
    Expects(maxDelta > 0);
    Expects(maxDelta % deltaStep == 0);
    Expects(integrationStep > 0);
}

std::vector<TimeAveragedMSD> TimeAveragedMSDCalculator::calculate(const std::vector<Trajectory> &trajectories) {
    if (trajectories.empty())
        return {};
    Expects(this->maxDelta < trajectories.front().getSize());

    std::vector<TimeAveragedMSD> resultVector(trajectories.size());
    for (auto &result : resultVector)
        result = TimeAveragedMSD(this->maxDelta/this->deltaStep + 1, this->deltaStep, this->integrationStep);

    _OMP_PARALLEL_FOR
    for (std::size_t resultI = 0; resultI < trajectories.size(); resultI++) {
        auto &trajectory = trajectories[resultI];
        auto &result = resultVector[resultI];
        for (std::size_t deltaStepI{}; deltaStepI <= this->maxDelta/this->deltaStep; deltaStepI++) {
            std::size_t delta = deltaStepI*this->deltaStep;
            float r2{};
            for (std::size_t trajStepI = delta; trajStepI < trajectory.getSize(); trajStepI++) {
                Move deltaR = trajectory[trajStepI] - trajectory[trajStepI - delta];

                r2 += deltaR.x*deltaR.x + deltaR.y*deltaR.y;
            }
            r2 /= (trajectory.getSize() - delta);
            result[deltaStepI].delta2 = r2;
        }
    }

    return resultVector;
}
