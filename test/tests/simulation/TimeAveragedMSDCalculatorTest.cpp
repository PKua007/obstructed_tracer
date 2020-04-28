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
    TimeAveragedMSDCalculator calculator(4, 2);

    TimeAveragedMSD tamsd = calculator.calculate(trajectory);

    REQUIRE_FALSE(tamsd.empty());
    REQUIRE(tamsd.size() == 3);
    REQUIRE(tamsd.getStepSize() == 2);
    REQUIRE(tamsd[0] == 0);
    REQUIRE(tamsd[1] == Approx(66.5));
    REQUIRE(tamsd[2] == Approx(39));
}
