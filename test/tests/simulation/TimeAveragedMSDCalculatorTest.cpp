/*
 * TimeAveragedMSDCalculatorTest.cpp
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "simulation/TimeAveragedMSDCalculator.h"

TEST_CASE("TimeAveragedMSDCalculator") {
    Trajectory trajectory;
    trajectory.addPoint({1, 2}, false);
    trajectory.addPoint({-4, 2}, false);
    trajectory.addPoint({1, 3}, false);
    trajectory.addPoint({4, 10}, false);
    trajectory.addPoint({5, -3}, false);
    trajectory.addPoint({2, 1}, false);
    std::size_t maxDelta = 4;
    std::size_t deltaStep = 2;
    float integrationStep = 0.25;
    TimeAveragedMSDCalculator calculator(maxDelta, deltaStep, integrationStep);

    TimeAveragedMSD tamsd = calculator.calculate(trajectory);

    // Assert trajectory "metadata"
    REQUIRE_FALSE(tamsd.empty());
    REQUIRE(tamsd.size() == 3);
    REQUIRE(tamsd.dataIndexToRealTime(1) == Approx(0.5));
    // Assert the actual data
    REQUIRE(tamsd[0] == 0);
    REQUIRE(tamsd[1] == Approx(66.5));
    REQUIRE(tamsd[2] == Approx(39));
}
