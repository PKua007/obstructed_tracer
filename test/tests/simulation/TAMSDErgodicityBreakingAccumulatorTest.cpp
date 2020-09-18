/*
 * TAMSDErgodicityBreakingAccumulatorTest.cpp
 *
 *  Created on: 18 wrz 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "matchers/VectorApproxEqualMatcher.h"

#include "simulation/TAMSDErgodicityBreakingAccumulator.h"

TEST_CASE("TAMSDErgodicityBreakingAccumulator") {
    // Step size 2, integration step 0.25, which gives delta_time = 0.5
    TAMSDErgodicityBreakingAccumulator accumulator(3, 2, 0.25);
    TimeAveragedMSD tamsd1(3, 2, 0.25);
    TimeAveragedMSD tamsd2(3, 2, 0.25);
    tamsd1[0].delta2 = 0;   tamsd1[1].delta2 = 5;   tamsd1[2].delta2 = 8;
    tamsd2[0].delta2 = 0;   tamsd2[1].delta2 = 3;   tamsd2[2].delta2 = 2;

    accumulator.addTAMSD(tamsd1);
    accumulator.addTAMSD(tamsd2);

    SECTION("EB parameters") {
        std::vector<double> EBs = accumulator.getEBParameters();

        REQUIRE_THAT(EBs, IsApproxEqual(std::vector<double>{0, 0.0625, 0.36}, 1e-8));
    }

    SECTION("storing") {
        std::ostringstream out;

        accumulator.storeEBParameters(out);

        REQUIRE(out.str() == "0 0\n0.5 0.0625\n1 0.36\n");
    }
}
