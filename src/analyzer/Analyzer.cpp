/*
 * Analyzer.cpp
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#include "Analyzer.h"
#include "PowerRegression.h"
#include "utils/Assertions.h"
#include <iostream>

void Analyzer::analyze(const MSDData &msdData) {
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
}
