/*
 * TimeAveragedMSDCalculator.cpp
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#include "TimeAveragedMSDCalculator.h"
#include "utils/Assertions.h"

TimeAveragedMSDCalculator::TimeAveragedMSDCalculator(std::size_t maxDelta, std::size_t deltaStep)
        : maxDelta{maxDelta}, deltaStep{deltaStep}
{
    Expects(maxDelta > 0);
    Expects(maxDelta % deltaStep == 0);
}

TimeAveragedMSD TimeAveragedMSDCalculator::calculate(const Trajectory &trajectory) {
    Expects(this->maxDelta < trajectory.getSize());

    TimeAveragedMSD result(this->maxDelta/this->deltaStep + 1, this->deltaStep);

    for (std::size_t step{}; step <= this->maxDelta/this->deltaStep; step++) {
        std::size_t delta = step*this->deltaStep;
        float r2{};
        for (std::size_t i = delta; i < trajectory.getSize(); i++) {
            Move deltaR = trajectory[i - delta] - trajectory[i];

            r2 += deltaR.x*deltaR.x + deltaR.y*deltaR.y;
        }
        r2 /= (trajectory.getSize() - delta);
        result[step] = r2;
    }

    return result;
}
