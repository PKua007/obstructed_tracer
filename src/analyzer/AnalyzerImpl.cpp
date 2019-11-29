/*
 * Analyzer.cpp
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#include <iostream>
#include <cmath>

#include "AnalyzerImpl.h"
#include "PowerRegression.h"
#include "utils/Assertions.h"

void AnalyzerImpl::analyze(const MSDData &msdData) {
    std::size_t trajectorySize = msdData.size();
    std::size_t startIndex = static_cast<std::size_t>(trajectorySize*this->relativeRangeStart);
    std::size_t endIndex = static_cast<std::size_t>(trajectorySize*this->relativeRangeEnd);
    Assert(startIndex > 0);
    Assert(endIndex <= trajectorySize);
    Assert(startIndex < endIndex);

    PowerRegression regression;
    for (std::size_t i = startIndex; i < endIndex; i++)
        regression.addXY(i, msdData[i].x2 + msdData[i].y2);
    regression.calculate();

    this->rSquareResult.D = regression.getMultiplier();
    this->rSquareResult.alpha = regression.getExponent();
    this->rSquareResult.R2 = regression.getR2();

    regression.clear();
    for (std::size_t i = startIndex; i < endIndex; i++)
        regression.addXY(i, msdData[i].x2 - msdData[i].x*msdData[i].x + msdData[i].y2 - msdData[i].y*msdData[i].y);
    regression.calculate();

    this->rVarianceResult.D = regression.getMultiplier();
    this->rVarianceResult.alpha = regression.getExponent();
    this->rVarianceResult.R2 = regression.getR2();

    std::size_t middleIndex = static_cast<size_t>(std::exp(std::log(static_cast<double>(trajectorySize)) / 2.));
    Assert(middleIndex >= 0 && middleIndex < trajectorySize);
    this->lastPointCorrelation = this->calculateCorrelation(msdData[trajectorySize - 1]);
    this->middlePointCorrelation = this->calculateCorrelation(msdData[middleIndex]);
}

double AnalyzerImpl::calculateCorrelation(MSDData::Entry msdEntry) {
    double covXY = msdEntry.xy - msdEntry.x*msdEntry.y;
    double varX = msdEntry.x2 - msdEntry.x*msdEntry.x;
    double varY = msdEntry.y2 - msdEntry.y*msdEntry.y;

    return covXY/std::sqrt(varX*varY);
}
