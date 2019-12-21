/*
 * TrajectoryTest.cpp
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <sstream>

#include "utils/CudaCheck.h"
#include "simulation/Trajectory.h"

TEST_CASE("Trajectory: adding") {
    SECTION("empty") {
        Trajectory trajectory;

        REQUIRE(trajectory.getSize() == 0);
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 0);
    }

    SECTION("simple") {
        Trajectory trajectory;

        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, false);
        trajectory.addPoint(Point{4, 5}, false);

        REQUIRE(trajectory.getSize() == 3);
        REQUIRE(trajectory[0] == Point{0, 1});
        REQUIRE(trajectory[1] == Point{2, 3});
        REQUIRE(trajectory[2] == Point{4, 5});
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 0);
    }

    SECTION("one accepted") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, true);
        trajectory.addPoint(Point{4, 5}, false);

        REQUIRE(trajectory.getSize() == 3);
        REQUIRE(trajectory[0] == Point{0, 1});
        REQUIRE(trajectory[1] == Point{2, 3});
        REQUIRE(trajectory[2] == Point{4, 5});
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 1);
    }

    SECTION("first accepted should throw") {
        Trajectory trajectory;

        REQUIRE_THROWS(trajectory.addPoint(Point{2, 3}, true));
    }

    SECTION("clear") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, true);
        trajectory.addPoint(Point{4, 5}, false);

        trajectory.clear();

        REQUIRE(trajectory.getSize() == 0);
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 0);
    }
}

TEST_CASE("Trajectory: access") {
    SECTION("first") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, false);
        trajectory.addPoint(Point{4, 5}, false);

        REQUIRE(trajectory.getFirst() == Point{0, 1});
    }

    SECTION("last") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, false);
        trajectory.addPoint(Point{4, 5}, false);

        REQUIRE(trajectory.getLast() == Point{4, 5});
    }
}

TEST_CASE("Trajectory: append") {
    SECTION("easy") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, false);
        trajectory.addPoint(Point{4, 5}, false);
        Trajectory another;
        another.addPoint(Point{4, 5}, false);
        another.addPoint(Point{6, 7}, false);

        trajectory.appendAnotherTrajectory(another);

        REQUIRE(trajectory.getSize() == 4);
        REQUIRE(trajectory[0] == Point{0, 1});
        REQUIRE(trajectory[1] == Point{2, 3});
        REQUIRE(trajectory[2] == Point{4, 5});
        REQUIRE(trajectory[3] == Point{6, 7});
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 0);
    }

    SECTION("accepted steps should add") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, true);
        trajectory.addPoint(Point{4, 5}, false);
        Trajectory another;
        another.addPoint(Point{4, 5}, false);
        another.addPoint(Point{6, 7}, true);

        trajectory.appendAnotherTrajectory(another);

        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 2);
    }

    SECTION("appending on empty") {
        Trajectory another;
        another.addPoint(Point{0, 1}, false);
        another.addPoint(Point{2, 3}, true);
        another.addPoint(Point{4, 5}, false);
        Trajectory trajectory;

        trajectory.appendAnotherTrajectory(another);

        REQUIRE(trajectory.getSize() == 3);
        REQUIRE(trajectory[0] == Point{0, 1});
        REQUIRE(trajectory[1] == Point{2, 3});
        REQUIRE(trajectory[2] == Point{4, 5});
        REQUIRE(trajectory.getNumberOfAcceptedSteps() == 1);
    }

    SECTION("non-equal last and first steps should throw") {
        Trajectory trajectory;
        trajectory.addPoint(Point{0, 1}, false);
        trajectory.addPoint(Point{2, 3}, false);
        Trajectory another;
        another.addPoint(Point{4, 5}, false);

        REQUIRE_THROWS(trajectory.appendAnotherTrajectory(another));
    }
}

TEST_CASE("Trajectory: storing") {
    Trajectory trajectory;
    trajectory.addPoint(Point{0, 1}, false);
    trajectory.addPoint(Point{2, 3}, true);
    trajectory.addPoint(Point{4, 5}, false);
    std::ostringstream out;

    trajectory.store(out);

    std::ostringstream outExpected;
    outExpected << "0 1" << std::endl << "2 3" << std::endl << "4 5" << std::endl;
    REQUIRE(out.str() == outExpected.str());
}

TEST_CASE("Trajectory: copy from GPU") {
    Trajectory trajectory;
    trajectory.addPoint(Point{0, 1}, false);
    trajectory.addPoint(Point{2, 3}, true);
    trajectory.addPoint(Point{4, 5}, true);
    std::vector<Point> cpuVector = {{6, 7}, {8, 9}};
    Point *gpuVector;
    cudaCheck( cudaMalloc(&gpuVector, sizeof(Point) * 2) );
    cudaCheck( cudaMemcpy(gpuVector, cpuVector.data(), sizeof(Point) * 2, cudaMemcpyHostToDevice) );

    trajectory.copyGPUData(gpuVector, 2, 1);
    cudaCheck( cudaFree(gpuVector) );

    REQUIRE(trajectory.getSize() == 2);
    REQUIRE(trajectory[0] == Point{6, 7});
    REQUIRE(trajectory[1] == Point{8, 9});
    REQUIRE(trajectory.getNumberOfAcceptedSteps() == 1);
}
