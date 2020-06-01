/*
 * SurvivalProbabilityAccumulatorTest.cpp
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "simulation/SurvivalProbabilityAccumulator.h"

TEST_CASE("SurvivalProbabilityAccumulator") {
    Trajectory traj1(10), traj2(10);
    traj1.addPoint({0, 0});
    traj1.addPoint({1, 1});
    traj1.addPoint({-1, 1});
    traj1.addPoint({-4, 3});
    traj1.addPoint({-1, -1});
    traj1.addPoint({-2, -3});
    traj1.addPoint({-1, -5});
    traj1.addPoint({-3, -6});
    traj1.addPoint({-4, -4});
    traj1.addPoint({-7, -4});
    traj1.addPoint({-7, -3});

    traj2.addPoint({0, 0});
    traj2.addPoint({2, -1});
    traj2.addPoint({1, -1});
    traj2.addPoint({1, -3});
    traj2.addPoint({3, -1});
    traj2.addPoint({5, -1});
    traj2.addPoint({5, 1});
    traj2.addPoint({3, 1});
    traj2.addPoint({2, 2});
    traj2.addPoint({2, 3});
    traj2.addPoint({-1, 3});

    SECTION("full test, each step checked") {
        SurvivalProbabilityAccumulator accumulator({2, 4, 6}, 10, 1, 3);

        accumulator.addTrajectories({traj1, traj2});

        std::vector<SurvivalProbability> sps = accumulator.calculateSurvivalProbabilities();

        REQUIRE(sps.size() == 3);
        std::vector<double> sp2(sps[0].begin(), sps[0].end()), sp4(sps[1].begin(), sps[1].end()),
                            sp6(sps[2].begin(), sps[2].end());
        REQUIRE(sps[0].getRadius() == 2);
        REQUIRE(sp2 == std::vector<double>{0.5, 0.5,   0,   0, 0, 0,   0,   0,   0,   0});
        REQUIRE(sps[1].getRadius() == 4);
        REQUIRE(sp4 == std::vector<double>{  1,   1, 0.5, 0.5, 0, 0,   0,   0,   0,   0});
        REQUIRE(sps[2].getRadius() == 6);
        REQUIRE(sp6 == std::vector<double>{  1,   1,   1,   1, 1, 1, 0.5, 0.5, 0.5, 0.5});
    }

    SECTION("every second step for radius = 4") {
        SurvivalProbabilityAccumulator accumulator({4}, 10, 2, 3);

        accumulator.addTrajectories({traj1, traj2});

        std::vector<SurvivalProbability> sps = accumulator.calculateSurvivalProbabilities();

        REQUIRE(sps.size() == 1);
        std::vector<double> sp(sps[0].begin(), sps[0].end());
        REQUIRE(sps[0].getRadius() == 4);
        REQUIRE(sp == std::vector<double>{1, 1, 0, 0, 0});
    }
}
