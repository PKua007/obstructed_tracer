/*
 * TAMSDPowerLawAccumulatorTest.cpp
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>

#include "matchers/VectorApproxEqualMatcher.h"

#include "simulation/TAMSDPowerLawAccumulator.h"

TEST_CASE("TAMSDPowerLawAccumulator") {
    // We will have steps 0, 1, 2, 3, 4, 5
    // The fit range (0.4, 0.9) will give 2, 3, 4
    TAMSDPowerLawAccumulator accumulator(0.4, 0.9);

    // Step size 2, integration step 0.25, which gives delta_time = 0.5
    TimeAveragedMSD tamsd1(6, 2, 0.25);
    TimeAveragedMSD tamsd2(6, 2, 0.25);

    // Now we fill only 2, 3, 4 steps with a nice power law, the rest with junk
    std::fill(tamsd1.begin(), tamsd1.end(), TimeAveragedMSD::Entry{1337, {1, 2}});
    std::fill(tamsd2.begin(), tamsd2.end(), TimeAveragedMSD::Entry{1337, {3, 4}});

    // delta2 is calculated from Kx * t^alphax
    const float K1 = 2, K2 = 3, alpha1 = 2, alpha2 = 4;
    tamsd1[2].delta2 = K1 * std::pow(2 * 0.25, alpha1);
    tamsd1[3].delta2 = K1 * std::pow(3 * 0.25, alpha1);
    tamsd1[4].delta2 = K1 * std::pow(4 * 0.25, alpha1);
    tamsd2[2].delta2 = K2 * std::pow(2 * 0.25, alpha2);
    tamsd2[3].delta2 = K2 * std::pow(3 * 0.25, alpha2);
    tamsd2[4].delta2 = K2 * std::pow(4 * 0.25, alpha2);

    // delta is prepared in such a way with  nontrivial x, y components, that delta^2 = 0.25 delta2
    tamsd1[2].delta = Move{std::sqrt(tamsd1[2].delta2 / 2.f - 0.1f), std::sqrt(tamsd1[2].delta2 / 2.f + 0.1f)} / 2.f;
    tamsd1[3].delta = Move{std::sqrt(tamsd1[3].delta2 / 2.f - 0.1f), std::sqrt(tamsd1[3].delta2 / 2.f + 0.1f)} / 2.f;
    tamsd1[4].delta = Move{std::sqrt(tamsd1[4].delta2 / 2.f - 0.1f), std::sqrt(tamsd1[4].delta2 / 2.f + 0.1f)} / 2.f;
    tamsd2[2].delta = Move{std::sqrt(tamsd2[2].delta2 / 2.f - 0.001f), std::sqrt(tamsd2[2].delta2 / 2.f + 0.001f)} / 2.f;
    tamsd2[3].delta = Move{std::sqrt(tamsd2[3].delta2 / 2.f - 0.001f), std::sqrt(tamsd2[3].delta2 / 2.f + 0.001f)} / 2.f;
    tamsd2[4].delta = Move{std::sqrt(tamsd2[4].delta2 / 2.f - 0.001f), std::sqrt(tamsd2[4].delta2 / 2.f + 0.001f)} / 2.f;

    accumulator.addTAMSD(tamsd1);
    accumulator.addTAMSD(tamsd2);

    SECTION("alpha histogram") {
        auto alphas = accumulator.getExponentHistogram();
        REQUIRE_THAT(alphas, IsApproxEqual(std::vector<double>{alpha1, alpha2}, 1e-8));
    }

    SECTION("variance alpha histogram") {
        auto alphas = accumulator.getVarianceExponentHistogram();
        REQUIRE_THAT(alphas, IsApproxEqual(std::vector<double>{alpha1, alpha2}, 1e-8));
    }

    // The values for mean are from the tested class, kept for regression testing
    SECTION("mean alpha") {
        double meanAlpha = accumulator.getEnsembleAveragedExponent();
        REQUIRE(meanAlpha == Approx(2.8533724704));
    }

    // Same here
    SECTION("mean variance alpha") {
        double meanAlpha = accumulator.getEnsembleAveragedVarianceExponent();
        REQUIRE(meanAlpha == Approx(2.8267251239));
    }

    SECTION("mean TA MSD") {
        TimeAveragedMSD meanTamsd = accumulator.getEnsembleAveragedTAMSD();
        REQUIRE(meanTamsd.dataIndexToRealTime(1) == Approx(0.5));
        // We assume that if one value entry is correct, all of them are
        REQUIRE(meanTamsd[2].delta2 == Approx((tamsd1[2].delta2 + tamsd2[2].delta2) / 2));
        REQUIRE(meanTamsd[2].delta.x == Approx((tamsd1[2].delta.x + tamsd2[2].delta.x) / 2));
        REQUIRE(meanTamsd[2].delta.y == Approx((tamsd1[2].delta.y + tamsd2[2].delta.y) / 2));
    }
}
