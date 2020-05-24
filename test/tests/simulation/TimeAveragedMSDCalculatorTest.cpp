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
    // So we have (delta in steps, actual delta - time): (0, 0), (2, 0.5), (4, 1)
    TimeAveragedMSDCalculator calculator(maxDelta, deltaStep, integrationStep);

    auto tamsds = calculator.calculate({trajectory});
    REQUIRE(tamsds.size() == 1);
    auto &tamsd = tamsds.front();

    // Assert trajectory "metadata"
    REQUIRE_FALSE(tamsd.empty());
    REQUIRE(tamsd.size() == 3);
    REQUIRE(tamsd.dataIndexToRealTime(1) == Approx(0.5));
    // Assert the actual data
    REQUIRE(tamsd[0] == TimeAveragedMSD::Entry{0, {0, 0}});
    REQUIRE(tamsd[1] == TimeAveragedMSD::Entry{66.5, {2.5, -1.5}});
    REQUIRE(tamsd[2] == TimeAveragedMSD::Entry{39, {5, -3}});
}
