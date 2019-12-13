/*
 * Analyzer.cpp
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#include <cmath>

#include "AnalyzerImpl.h"
#include "PowerRegression.h"
#include "utils/Assertions.h"

void AnalyzerImpl::analyze(const MSDData &msdData, const Parameters &parameters, double relativeRangeStart,
                           double relativeRangeEnd)
{
    Expects(relativeRangeStart >= 0. && relativeRangeStart <= 1.);
    Expects(relativeRangeEnd >= 0. && relativeRangeEnd <= 1.);
    Expects(relativeRangeStart < relativeRangeEnd);

    std::size_t trajectorySize = msdData.size();
    std::size_t startIndex = static_cast<std::size_t>(trajectorySize*relativeRangeStart);
    std::size_t endIndex = static_cast<std::size_t>(trajectorySize*relativeRangeEnd);
    Assert(startIndex > 0);
    Assert(endIndex <= trajectorySize);
    Assert(startIndex < endIndex);

    PowerRegression regression;
    for (std::size_t i = startIndex; i < endIndex; i++)
        regression.addXY(i * parameters.integrationStep, msdData[i].x2 + msdData[i].y2);
    regression.calculate();

    this->rSquareResult.D = regression.getMultiplier();
    this->rSquareResult.alpha = regression.getExponent();
    this->rSquareResult.R2 = regression.getR2();

    regression.clear();
    for (std::size_t i = startIndex; i < endIndex; i++) {
        regression.addXY(i * parameters.integrationStep,
                         msdData[i].x2 - msdData[i].x*msdData[i].x + msdData[i].y2 - msdData[i].y*msdData[i].y);
    }
    regression.calculate();

    this->rVarianceResult.D = regression.getMultiplier();
    this->rVarianceResult.alpha = regression.getExponent();
    this->rVarianceResult.R2 = regression.getR2();

    // Middle point in log time scale - so index corresponding to sqrt(t_max)
    std::size_t middleIndex = static_cast<size_t>(std::sqrt(endIndex / parameters.integrationStep));
    Assert(middleIndex >= 0 && middleIndex < endIndex);
    this->lastPointCorrelation = this->calculateCorrelation(msdData[endIndex - 1]);
    this->middlePointCorrelation = this->calculateCorrelation(msdData[middleIndex]);
}

double AnalyzerImpl::calculateCorrelation(MSDData::Entry msdEntry) {
    double covXY = msdEntry.xy - msdEntry.x*msdEntry.y;
    double varX = msdEntry.x2 - msdEntry.x*msdEntry.x;
    double varY = msdEntry.y2 - msdEntry.y*msdEntry.y;

    return covXY/std::sqrt(varX*varY);
}
