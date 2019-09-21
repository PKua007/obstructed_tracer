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

Analyzer::Result Analyzer::analyze(const MSDData& msdData) {
    std::size_t trajectorySize = msdData.size();
    std::size_t startIndex = trajectorySize / 100;
    Assert(startIndex < trajectorySize - 1);

    PowerRegression regression;
    for (std::size_t i = startIndex; i < trajectorySize; i++)
        regression.addXY(i, msdData[i].x2 + msdData[i].y2);
    regression.calculate();

    Result result;
    result.D = {regression.getB(), 0};
    result.alpha = {regression.getA(), regression.getSA()};
    return result;
}
