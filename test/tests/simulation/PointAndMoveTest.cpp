/*
 * PointAndMoveTest.cpp
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <sstream>

#include "simulation/Point.h"
#include "simulation/Move.h"


TEST_CASE("Move") {
    SECTION("default construction") {
        Move m;

        REQUIRE(m == Move{0, 0});
    }

    SECTION("addition") {
        Move m1{1, 2};
        Move m2{3, 4};

        Move m = m1 + m2;
        REQUIRE(m == Move{4, 6});

        m += Move{-2, -3};
        REQUIRE(m == Move{2, 3});
    }

    SECTION("scalar multiplication") {
        Move m{1, 2};

        Move m1 = m * 2;
        Move m2 = 2 * m;

        REQUIRE(m1 == Move{2, 4});
        REQUIRE(m2 == Move{2, 4});
    }

    SECTION("scalar division") {
        Move m{6, 12};

        Move m1 = m / 2;
        REQUIRE(m1 == Move{3, 6});

        m1 /= 3;
        REQUIRE(m1 == Move{1, 2});
    }
}

TEST_CASE("Point") {
    SECTION("default construction") {
        Point p;

        REQUIRE(p == Point{0, 0});
    }

    SECTION("move addition") {
        Point p{1, 2};
        Move m{2, 4};

        Point p1 = p + m;

        REQUIRE(p1 == Point{3, 6});
    }

    SECTION("move addition assignment") {
        Point p{1, 2};
        Move m{2, 4};

        p += m;

        REQUIRE(p == Point{3, 6});
    }

    SECTION("move subtraction") {
        Point p{1, 2};
        Move m{-2, -4};

        Point p1 = p - m;

        REQUIRE(p1 == Point{3, 6});
    }

    SECTION("move subtraction assignment") {
        Point p{1, 2};
        Move m{-2, -4};

        p -= m;

        REQUIRE(p == Point{3, 6});
    }

    SECTION("not-equal operator") {
        REQUIRE_FALSE(Point{1, 2} != Point{1, 2});
        REQUIRE(Point{1, 2} != Point{1, 1});
        REQUIRE(Point{1, 2} != Point{2, 2});
    }

    SECTION("operator<<") {
        Point p{1, 2};
        std::ostringstream out;

        out << p;

        REQUIRE(out.str() == "1 2");
    }

    SECTION("2 points subtraction") {
        Point p1{1, 2};
        Point p2{7, 10};

        REQUIRE(p2 - p1 == Move{6, 8});
    }
}
