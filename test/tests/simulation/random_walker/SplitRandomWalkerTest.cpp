/*
 * SplitRandomWalkerTest.cpp
 *
 *  Created on: 22 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>
#include <sstream>

#include "simulation/random_walker/SplitRandomWalker.h"

#include "mocks/simulation/RandomWalkerMock.h"

TEST_CASE("SplitRandomWalker: basics") {
    SECTION("correct number of trajectories") {
        auto randomWalker = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        ALLOW_CALL_V(*randomWalker, getNumberOfTrajectories(),
            .RETURN(2));
        ALLOW_CALL_V(*randomWalker, getNumberOfSteps(),
            .RETURN(3));

        SplitRandomWalker splitRandomWalker(4, std::move(randomWalker));

        REQUIRE(splitRandomWalker.getNumberOfTrajectories() == 2);
    }

    SECTION("correct total number of steps") {
        auto randomWalker = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        ALLOW_CALL_V(*randomWalker, getNumberOfTrajectories(),
            .RETURN(2));
        ALLOW_CALL_V(*randomWalker, getNumberOfSteps(),
            .RETURN(3));

        SplitRandomWalker splitRandomWalker(4, std::move(randomWalker));

        REQUIRE(splitRandomWalker.getNumberOfSteps() == 12);
    }
}

TEST_CASE("SplitRandomWalker: 2 splits, 1 trajectory") {
    Trajectory t1;
    t1.addPoint(Point{1, 2});
    t1.addPoint(Point{3, 4});
    Trajectory t2;
    t2.addPoint(Point{3, 4});
    t2.addPoint(Point{5, 6});
    std::vector<Tracer> initialTracers1 = {Tracer{Point{1, 2}, 0}};
    std::vector<Tracer> initialTracers2 = {Tracer{Point{3, 4}, 0}};

    auto randomWalker = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
    ALLOW_CALL_V(*randomWalker, getNumberOfTrajectories(),
        .RETURN(1));
    ALLOW_CALL_V(*randomWalker, getNumberOfSteps(),
        .RETURN(1));

    trompeloeil::sequence seq;
    using trompeloeil::_;

    // First split
    REQUIRE_CALL_V(*randomWalker, run(_, initialTracers1),
        .IN_SEQUENCE(seq));
    REQUIRE_CALL_V(*randomWalker, getTrajectory(0),
        .IN_SEQUENCE(seq)
        .TIMES(AT_LEAST(1))
        .LR_RETURN((t1)));

    // Second split
    REQUIRE_CALL_V(*randomWalker, run(_, initialTracers2),
        .IN_SEQUENCE(seq));
    REQUIRE_CALL_V(*randomWalker, getTrajectory(0),
        .IN_SEQUENCE(seq)
        .TIMES(AT_LEAST(1))
        .LR_RETURN((t2)));

    SplitRandomWalker splitRandomWalker(2, std::move(randomWalker));


    std::ostringstream logger;
    splitRandomWalker.run(logger, initialTracers1);


    REQUIRE(splitRandomWalker.getNumberOfTrajectories() == 1);
    auto actualTrajectory = splitRandomWalker.getTrajectory(0);
    REQUIRE(actualTrajectory.getSize() == 3);
    REQUIRE(actualTrajectory[0] == Point{1, 2});
    REQUIRE(actualTrajectory[1] == Point{3, 4});
    REQUIRE(actualTrajectory[2] == Point{5, 6});
}
