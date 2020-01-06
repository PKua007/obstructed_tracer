/*
 * CPURandomWalkerTest.cpp
 *
 *  Created on: 22 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "simulation/random_walker/CPURandomWalker.h"

#include "mocks/simulation/MoveFilterMock.h"
#include "mocks/simulation/MoveGeneratorMock.h"

using Catch::Contains;

TEST_CASE("CPURandomWalker: basic") {
    RandomWalker::WalkParameters walkParameters;
    walkParameters.numberOfSteps = 1000;
    walkParameters.tracerRadius = 3;
    walkParameters.integrationStep = 0.1;
    walkParameters.drift = Move{1, 2};

    auto moveFilter = std::unique_ptr<MoveFilterMock>(new MoveFilterMock);
    auto moveGenerator = std::unique_ptr<MoveGeneratorMock>(new MoveGeneratorMock);

    std::ostringstream logger;

    SECTION("correct number of trajectories") {
        ALLOW_CALL_V(*moveFilter, setupForTracerRadius(ANY(float)));
        CPURandomWalker cpuRandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);

        REQUIRE(cpuRandomWalker.getNumberOfTrajectories() == 5);
    }

    SECTION("correct number of steps") {
        ALLOW_CALL_V(*moveFilter, setupForTracerRadius(ANY(float)));
        CPURandomWalker cpuRandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);

        REQUIRE(cpuRandomWalker.getNumberOfSteps() == 1000);
    }

    SECTION("correct tracer radius") {
        REQUIRE_CALL_V(*moveFilter, setupForTracerRadius(3));

        CPURandomWalker cpuRandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);
    }

    SECTION("incorrect number of trajectories should throw") {
        REQUIRE_THROWS_WITH(
            CPURandomWalker(0, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains("> 0")
        );
    }

    SECTION("incorrect number of steps should throw") {
        walkParameters.numberOfSteps = 0;

        REQUIRE_THROWS_WITH(
            CPURandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains("> 0")
        );
    }

    SECTION("incorrect tracer radius should throw") {
        walkParameters.tracerRadius = -0.1;

        REQUIRE_THROWS_WITH(
            CPURandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains(">= 0")
        );
    }

    SECTION("tracer radius can be 0") {
        walkParameters.tracerRadius = 0;
        ALLOW_CALL_V(*moveFilter, setupForTracerRadius(ANY(float)),
            .THROW(std::runtime_error("tracer radius setup reached")));

        REQUIRE_THROWS_WITH(
            CPURandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains("tracer radius setup reached")
        );
    }

    SECTION("incorrect integration step should throw") {
        walkParameters.integrationStep = 0;

        REQUIRE_THROWS_WITH(
            CPURandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains("> 0")
        );

        walkParameters.integrationStep = -0.1;

        REQUIRE_THROWS_WITH(
            CPURandomWalker(5, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger),
            Contains("> 0")
        );
    }
}

TEST_CASE("CPURandomWalker: sampling valid tracers") {
    RandomWalker::WalkParameters walkParameters;
    walkParameters.numberOfSteps = 1000;
    walkParameters.tracerRadius = 3;
    walkParameters.integrationStep = 0.1;
    walkParameters.drift = Move{1, 2};

    auto moveFilter = std::unique_ptr<MoveFilterMock>(new MoveFilterMock);
    REQUIRE_CALL_V(*moveFilter, setupForTracerRadius(3));
    trompeloeil::sequence seq;
    REQUIRE_CALL_V(*moveFilter, randomValidTracer(),
        .IN_SEQUENCE(seq)
        .RETURN(Tracer{Point{1, 2}, 3}));
    REQUIRE_CALL_V(*moveFilter, randomValidTracer(),
        .IN_SEQUENCE(seq)
        .RETURN(Tracer{Point{3, 4}, 3}));
    REQUIRE_CALL_V(*moveFilter, randomValidTracer(),
        .IN_SEQUENCE(seq)
        .RETURN(Tracer{Point{5, 6}, 3}));

    auto moveGenerator = std::unique_ptr<MoveGeneratorMock>(new MoveGeneratorMock);

    std::ostringstream logger;

    CPURandomWalker cpuRandomWalker(3, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);


    auto initialTracers = cpuRandomWalker.getRandomInitialTracersVector();


    REQUIRE(initialTracers.size() == 3);
    REQUIRE(initialTracers[0] == Tracer{Point{1, 2}, 3});
    REQUIRE(initialTracers[1] == Tracer{Point{3, 4}, 3});
    REQUIRE(initialTracers[2] == Tracer{Point{5, 6}, 3});
}

TEST_CASE("CPURandomWalker: run (1 trajectory)") {
    RandomWalker::WalkParameters walkParameters;
    walkParameters.numberOfSteps = 2;
    walkParameters.tracerRadius = 3;
    walkParameters.integrationStep = 1;
    walkParameters.drift = Move{1, 2};

    auto moveFilter = std::unique_ptr<MoveFilterMock>(new MoveFilterMock);
    REQUIRE_CALL_V(*moveFilter, setupForTracerRadius(3));

    auto moveGenerator = std::unique_ptr<MoveGeneratorMock>(new MoveGeneratorMock);
    trompeloeil::sequence seq;
    REQUIRE_CALL_V(*moveGenerator, generateMove(),
        .IN_SEQUENCE(seq)
        .RETURN(Move{3, 4}));
    REQUIRE_CALL_V(*moveGenerator, generateMove(),
        .IN_SEQUENCE(seq)
        .RETURN(Move{5, 6}));

    std::ostringstream logger;

    SECTION("all steps accepted") {
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{1, 2}, 3}, Move{4, 6}),
            .RETURN(true));
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{5, 8}, 3}, Move{6, 8}),
            .RETURN(true));
        CPURandomWalker cpuRandomWalker(1, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);

        cpuRandomWalker.run(logger, std::vector<Tracer>{Tracer{Point{1, 2}, 3}});

        REQUIRE(cpuRandomWalker.getNumberOfTrajectories() == 1);
        auto traj = cpuRandomWalker.getTrajectory(0);
        REQUIRE(traj.getSize() == 3);
        REQUIRE(traj[0] == Point{1, 2});
        REQUIRE(traj[1] == Point{5, 8});
        REQUIRE(traj[2] == Point{11, 16});
        REQUIRE(traj.getNumberOfAcceptedSteps() == 2);
    }

    SECTION("1 steps rejected") {
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{1, 2}, 3}, Move{4, 6}),
            .RETURN(false));
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{1, 2}, 3}, Move{6, 8}),
            .RETURN(true));
        CPURandomWalker cpuRandomWalker(1, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);

        cpuRandomWalker.run(logger, std::vector<Tracer>{Tracer{Point{1, 2}, 3}});

        REQUIRE(cpuRandomWalker.getNumberOfTrajectories() == 1);
        auto traj = cpuRandomWalker.getTrajectory(0);
        REQUIRE(traj.getSize() == 3);
        REQUIRE(traj[0] == Point{1, 2});
        REQUIRE(traj[1] == Point{1, 2});
        REQUIRE(traj[2] == Point{7, 10});
        REQUIRE(traj.getNumberOfAcceptedSteps() == 1);
    }

    SECTION("integration step rescaling drift") {
        walkParameters.integrationStep = 2;
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{1, 2}, 3}, Move{5, 8}),
            .RETURN(true));
        REQUIRE_CALL_V(*moveFilter, isMoveValid(Tracer{Point{6, 10}, 3}, Move{7, 10}),
            .RETURN(true));
        CPURandomWalker cpuRandomWalker(1, walkParameters, std::move(moveGenerator), std::move(moveFilter), logger);

        cpuRandomWalker.run(logger, std::vector<Tracer>{Tracer{Point{1, 2}, 3}});

        REQUIRE(cpuRandomWalker.getNumberOfTrajectories() == 1);
        auto traj = cpuRandomWalker.getTrajectory(0);
        REQUIRE(traj.getSize() == 3);
        REQUIRE(traj[0] == Point{1, 2});
        REQUIRE(traj[1] == Point{6, 10});
        REQUIRE(traj[2] == Point{13, 20});
        REQUIRE(traj.getNumberOfAcceptedSteps() == 2);
    }
}
