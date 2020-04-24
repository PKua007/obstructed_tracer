/*
 * CoverageMapAccumulatorTest.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "simulation/CoverageMapAccumulator.h"

TEST_CASE("CoverageMapAccumulator: 10x10 map") {
    Trajectory trajectory1;
    trajectory1.addPoint(Point{0.2, 1.0}, false);  // #1
    trajectory1.addPoint(Point{2.1, 3.9}, true);  // #2
    trajectory1.addPoint(Point{0.9, 1.5}, true);  // #3, same as #1, but with different fractional part
    Trajectory trajectory2;
    trajectory2.addPoint(Point{3.2, 1.0}, false);
    trajectory2.addPoint(Point{5.5, 2.9}, true);
    trajectory2.addPoint(Point{10.4, 1.5}, true);  // same as #1 and #3 from the previous, but w.r.t. pbc
    CoverageMapAccumulator accumulator(10, 10);

    // Do in 2 batches to test this as well
    accumulator.addTrajectories({trajectory1});
    accumulator.addTrajectories({trajectory2});

    auto map = accumulator.getCoverageMap();
    REQUIRE(map.numOfVisits({0, 1}) == 3);
    REQUIRE(map.numOfVisits({2, 3}) == 1);
    REQUIRE(map.numOfVisits({3, 1}) == 1);
    REQUIRE(map.numOfVisits({5, 2}) == 1);
    REQUIRE(map.numOfVisits({2, 6}) == 0);
    auto mapCountedOnce = accumulator.getCoverageMapCountedOnce();
    REQUIRE(mapCountedOnce.numOfVisits({0, 1}) == 2);   // because in the first one this point was visited 2 times
    REQUIRE(mapCountedOnce.numOfVisits({2, 3}) == 1);
    REQUIRE(mapCountedOnce.numOfVisits({3, 1}) == 1);
    REQUIRE(mapCountedOnce.numOfVisits({5, 2}) == 1);
    REQUIRE(mapCountedOnce.numOfVisits({2, 6}) == 0);
}
