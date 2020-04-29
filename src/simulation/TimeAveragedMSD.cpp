/*
 * TimeAveragedMSD.cpp
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#include <algorithm>

#include "TimeAveragedMSD.h"

#include "utils/Assertions.h"
#include "analyzer/PowerRegression.h"

TimeAveragedMSD::TimeAveragedMSD(std::size_t numSteps, std::size_t stepSize, float integrationStep)
            : stepSize{stepSize}, data(numSteps), integrationStep(integrationStep)
{
    Expects(stepSize > 0);
    Expects(numSteps > 0);
    Expects(integrationStep > 0);
}

float &TimeAveragedMSD::operator[](std::size_t stepIdx) {
    Expects(stepIdx < this->size());
    return data[stepIdx];
}

float TimeAveragedMSD::operator[](std::size_t stepIdx) const {
    Expects(stepIdx < this->size());
    return data[stepIdx];
}

double TimeAveragedMSD::getPowerLawExponent(double relativeFitStart, double relativeFitEnd) const {
    Expects(relativeFitStart > 0);
    Expects(relativeFitEnd > relativeFitStart);
    Expects(relativeFitEnd <= 1);

    std::size_t fitStartIdx = static_cast<std::size_t>(this->size()*relativeFitStart);
    std::size_t fitEndIdx = static_cast<std::size_t>(this->size()*relativeFitEnd);
    Assert(fitStartIdx > 0);    // TA MSD is 0 for Delta=0, so it breaks the logarithm in the power-law fit
    Assert(fitEndIdx < this->size());

    PowerRegression regression;
    for (std::size_t i = fitStartIdx; i < fitEndIdx; i++)
        regression.addXY(this->dataIndexToRealTime(i), this->data[i]);
    regression.calculate();
    return regression.getExponent().value;
}

TimeAveragedMSD &TimeAveragedMSD::operator+=(const TimeAveragedMSD &other) {
    Expects(this->size() == other.size());
    Expects(this->stepSize == other.stepSize);
    Expects(this->integrationStep == other.integrationStep);

    std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::plus<float>());
    return *this;
}

TimeAveragedMSD operator/(const TimeAveragedMSD &tamsd, float a) {
    TimeAveragedMSD result(tamsd.size(), tamsd.stepSize, tamsd.integrationStep);
    for (std::size_t i{}; i < tamsd.size(); i++)
        result[i] = tamsd.data[i] / a;
    return result;
}
