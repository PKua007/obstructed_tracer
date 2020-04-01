/*
 * PositionHistogramTest.cpp
 *
 *  Created on: 1 kwi 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "simulation/PositionHistogram.h"

#include "mocks/simulation/RandomWalkerMock.h"

TEST_CASE("PositionHistogram: valid") {
    Trajectory trajectory1;
    trajectory1.addPoint(Point{0, 1}, false);
    trajectory1.addPoint(Point{3, 6}, false);
    trajectory1.addPoint(Point{10, 7}, false);

    Trajectory trajectory2;
    trajectory2.addPoint(Point{1, 2}, false);
    trajectory2.addPoint(Point{5, 8}, false);
    trajectory2.addPoint(Point{6, 9}, false);

    SECTION("one walker, two trajectories") {
        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(2)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(1),
            .RETURN(trajectory2)
            .TIMES(AT_LEAST(1)));
        PositionHistogram histogram({0, 2});

        histogram.addTrajectories(randomWalker);

        std::ostringstream out0, out2;
        histogram.printForStep(0, out0);
        histogram.printForStep(2, out2);
        REQUIRE(out0.str() == "0 1\n1 2\n");
        REQUIRE(out2.str() == "10 7\n6 9\n");
    }

    SECTION("two walker, one trajectory each") {
        RandomWalkerMock randomWalker1;
        REQUIRE_CALL_V(randomWalker1, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker1, getTrajectory(0),
            .RETURN(trajectory1)
            .TIMES(AT_LEAST(1)));
        RandomWalkerMock randomWalker2;
        REQUIRE_CALL_V(randomWalker2, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker2, getTrajectory(0),
            .RETURN(trajectory2)
            .TIMES(AT_LEAST(1)));
        PositionHistogram histogram({0, 2});

        histogram.addTrajectories(randomWalker1);
        histogram.addTrajectories(randomWalker2);

        std::ostringstream out0, out2;
        histogram.printForStep(0, out0);
        histogram.printForStep(2, out2);
        REQUIRE(out0.str() == "0 1\n1 2\n");
        REQUIRE(out2.str() == "10 7\n6 9\n");
    }
}

TEST_CASE("PositionHistogram: invalid") {
    Trajectory trajectory1;
    trajectory1.addPoint(Point{0, 1}, false);
    trajectory1.addPoint(Point{3, 6}, false);
    trajectory1.addPoint(Point{10, 7}, false);

    RandomWalkerMock randomWalker;
    ALLOW_CALL_V(randomWalker, getNumberOfTrajectories(),
        .RETURN(1));
    ALLOW_CALL_V(randomWalker, getTrajectory(0),
        .RETURN(trajectory1));
    PositionHistogram histogram({0, 3});

    SECTION("to big step in trajectory") {
        REQUIRE_THROWS(histogram.addTrajectories(randomWalker));
    }

    SECTION("to big step idx in printing") {
        std::ostringstream out;
        REQUIRE_THROWS(histogram.printForStep(2, out));
    }
}
