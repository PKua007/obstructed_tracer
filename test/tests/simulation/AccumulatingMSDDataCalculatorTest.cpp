/*
 * AccumulatingMSDDataCalculatorTest.cpp
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <trompeloeil_for_cuda/catch2/trompeloeil.hpp>

#include "simulation/AccumulatingMSDDataCalculator.h"

#include "mocks/simulation/RandomWalkerMock.h"
#include "matchers/frontend/MSDDataApproxEqualMatcher.h"

TEST_CASE("AccumulatingMSDDataCalculator: adding") {
    SECTION("one walker, one trajectory") {
        Trajectory trajectory;
        trajectory.addPoint(Point{1, 2}, false);
        trajectory.addPoint(Point{3, 5}, false);

        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory)
            .TIMES(AT_LEAST(1)));

        AccumulatingMSDDataCalculator calculator(1);


        calculator.addTrajectories(randomWalker);
        MSDData result = calculator.fetchMSDData();


        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl << "2 3 4 9 6" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-8));
    }

    SECTION("determining steps by the first trajectory") {
        Trajectory trajectory;
        trajectory.addPoint(Point{1, 2}, false);
        trajectory.addPoint(Point{3, 5}, false);

        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory)
            .TIMES(AT_LEAST(1)));

        AccumulatingMSDDataCalculator calculator;


        calculator.addTrajectories(randomWalker);
        MSDData result = calculator.fetchMSDData();


        REQUIRE(result.size() == 2);
    }

    SECTION("one walker, two trajectories") {
        Trajectory trajectory1;
        trajectory1.addPoint(Point{1, 2}, false);
        trajectory1.addPoint(Point{3, 6}, false);
        Trajectory trajectory2;
        trajectory2.addPoint(Point{1, 2}, false);
        trajectory2.addPoint(Point{5, 8}, false);

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

        AccumulatingMSDDataCalculator calculator(1);


        calculator.addTrajectories(randomWalker);
        MSDData result = calculator.fetchMSDData();


        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl << "3 5 10 26 16" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-8));
    }

    SECTION("two walker, one trajectory each") {
        Trajectory trajectory1;
        trajectory1.addPoint(Point{1, 2}, false);
        trajectory1.addPoint(Point{3, 6}, false);
        Trajectory trajectory2;
        trajectory2.addPoint(Point{1, 2}, false);
        trajectory2.addPoint(Point{5, 8}, false);

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

        AccumulatingMSDDataCalculator calculator(1);


        calculator.addTrajectories(randomWalker1);
        calculator.addTrajectories(randomWalker2);
        MSDData result = calculator.fetchMSDData();


        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl << "3 5 10 26 16" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-8));
    }
}

TEST_CASE("AccumulatinMSDDataCalculator: error check") {
    SECTION("not matching trajectory sizes") {
        Trajectory trajectory1;
        trajectory1.addPoint(Point{1, 2}, false);
        trajectory1.addPoint(Point{3, 6}, false);
        Trajectory trajectory2;
        trajectory2.addPoint(Point{1, 2}, false);
        trajectory2.addPoint(Point{5, 8}, false);
        trajectory2.addPoint(Point{2, 38}, false);

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

        AccumulatingMSDDataCalculator calculator(1);
        calculator.addTrajectories(randomWalker1);


        REQUIRE_THROWS(calculator.addTrajectories(randomWalker2));
    }

    SECTION("do not allow empty trajectories") {
        Trajectory trajectory;

        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory)
            .TIMES(AT_LEAST(1)));

        AccumulatingMSDDataCalculator calculator;


        REQUIRE_THROWS(calculator.addTrajectories(randomWalker));
    }

    SECTION("allow trivial (one-element) trajectories") {
        Trajectory trajectory;
        trajectory.addPoint(Point{1, 2}, false);

        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory)
            .TIMES(AT_LEAST(1)));

        AccumulatingMSDDataCalculator calculator(0);


        calculator.addTrajectories(randomWalker);
        MSDData result = calculator.fetchMSDData();


        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-8));
    }
}

TEST_CASE("AccumulatingMSDDataCalculator: fetching") {
    SECTION("fetching empty, initialized data") {
        AccumulatingMSDDataCalculator calculator(1);

        MSDData result = calculator.fetchMSDData();

        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl << "0 0 0 0 0" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result, IsApproxEqual(expected, 1e-8));
    }

    SECTION("fetching empty, uninitialized data") {
        AccumulatingMSDDataCalculator calculator;

        MSDData result = calculator.fetchMSDData();

        REQUIRE_THAT(result, IsApproxEqual(MSDData{}, 0));
    }

    SECTION("fetching clears data") {
        Trajectory trajectory;
        trajectory.addPoint(Point{1, 2}, false);
        trajectory.addPoint(Point{3, 5}, false);

        RandomWalkerMock randomWalker;
        REQUIRE_CALL_V(randomWalker, getNumberOfTrajectories(),
            .RETURN(1)
            .TIMES(AT_LEAST(1)));
        REQUIRE_CALL_V(randomWalker, getTrajectory(0),
            .RETURN(trajectory)
            .TIMES(AT_LEAST(1)));

        AccumulatingMSDDataCalculator calculator;
        calculator.addTrajectories(randomWalker);
        MSDData result = calculator.fetchMSDData();


        MSDData result2 = calculator.fetchMSDData();


        MSDData expected;
        std::stringstream expectedStream;
        expectedStream << "0 0 0 0 0" << std::endl << "0 0 0 0 0" << std::endl;
        expectedStream >> expected;
        REQUIRE_THAT(result2, IsApproxEqual(expected, 1e-8));
    }
}
