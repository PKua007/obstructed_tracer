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

Analyzer::Result Analyzer::analyze(const MSDData &msdData) {
    std::size_t trajectorySize = msdData.size();
    // For a while we hardcode calculating regression for 2 last orders of points
    std::size_t startIndex = trajectorySize / 100;
    Assert(startIndex > 0);
    Assert(startIndex < trajectorySize - 1);

    PowerRegression regression;
    for (std::size_t i = startIndex; i < trajectorySize; i++)
        regression.addXY(i, msdData[i].x2 + msdData[i].y2);
    regression.calculate();

    Result result;
    result.D = regression.getMultiplier();
    result.alpha = regression.getExponent();
    return result;
}
