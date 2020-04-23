/*
 * CoverageMapTest.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "simulation/CoverageMap.h"

TEST_CASE("CoverageMap: visiting") {
    SECTION("basic") {
        CoverageMap map(3, 7);

        map.visit({1, 2});
        map.visit({2, 4});
        map.visit({2, 4});

        REQUIRE(map.isVisited({1, 2}));
        REQUIRE(map.isVisited({2, 4}));
        REQUIRE_FALSE(map.isVisited({0, 2}));
        REQUIRE(map.numOfVisits({1, 2}) == 1);
        REQUIRE(map.numOfVisits({2, 4}) == 2);
        REQUIRE(map.numOfVisits({0, 2}) == 0);
    }

    SECTION("periodic") {
        CoverageMap map(10, 7);

        map.visit({1, 2});
        map.visit({1, 23});
        map.visit({-39, 2});
        map.visit({-39, 23});

        REQUIRE(map.isVisited({1, 2}));
        REQUIRE(map.isVisited({-39, 23}));
        REQUIRE(map.numOfVisits({1, 2}) == 4);
        REQUIRE(map.numOfVisits({-39, 23}) == 4);
    }
}

TEST_CASE("CoverageMap: adding two together") {
    CoverageMap map1(2, 2);
    CoverageMap map2(2, 2);
    map1.visit({1, 0});
    map1.visit({1, 0});
    map1.visit({0, 0});
    map2.visit({1, 1});
    map2.visit({1, 0});
    map2.visit({0, 1});

    auto map = map1 + map2;

    REQUIRE(map.numOfVisits({0, 0}) == 1);
    REQUIRE(map.numOfVisits({0, 1}) == 1);
    REQUIRE(map.numOfVisits({1, 0}) == 3);
    REQUIRE(map.numOfVisits({1, 1}) == 1);
}
