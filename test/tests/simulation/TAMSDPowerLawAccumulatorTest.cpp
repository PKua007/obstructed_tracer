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
    std::fill(tamsd1.begin(), tamsd1.end(), TimeAveragedMSD::Entry{1337, {0, 0}});
    std::fill(tamsd2.begin(), tamsd2.end(), TimeAveragedMSD::Entry{1337, {0, 0}});
    const float K1 = 2, K2 = 3, alpha1 = 2, alpha2 = 4;
    tamsd1[2].delta2 = K1 * std::pow(2 * 0.25, alpha1);
    tamsd1[3].delta2 = K1 * std::pow(3 * 0.25, alpha1);
    tamsd1[4].delta2 = K1 * std::pow(4 * 0.25, alpha1);
    tamsd2[2].delta2 = K2 * std::pow(2 * 0.25, alpha2);
    tamsd2[3].delta2 = K2 * std::pow(3 * 0.25, alpha2);
    tamsd2[4].delta2 = K2 * std::pow(4 * 0.25, alpha2);

    accumulator.addTAMSD(tamsd1);
    accumulator.addTAMSD(tamsd2);

    SECTION("alpha histogram") {
        auto alphas = accumulator.getExponentHistogram();
        REQUIRE_THAT(alphas, IsApproxEqual(std::vector<double>{alpha1, alpha2}, 1e-8));
    }

    SECTION("mean alpha") {
        double meanAlpha = accumulator.getEnsembleAveragedExponent();
        REQUIRE(meanAlpha == Approx(2.8533724704));
    }

    SECTION("mean TA MSD") {
        TimeAveragedMSD meanTamsd = accumulator.getEnsembleAveragedTAMSD();
        REQUIRE(meanTamsd.dataIndexToRealTime(1) == Approx(0.5));

        std::vector<float> meanTamsdVector(meanTamsd.size());
        std::transform(meanTamsd.begin(), meanTamsd.end(), meanTamsdVector.begin(), [](TimeAveragedMSD::Entry e) {
            return e.delta2;
        });
        REQUIRE_THAT(meanTamsdVector, IsApproxEqual(
            std::vector<float>{1337, 1337, 0.5f*(tamsd1[2].delta2+tamsd2[2].delta2), 0.5f*(tamsd1[3].delta2+tamsd2[3].delta2),
                                0.5f*(tamsd1[4].delta2+tamsd2[4].delta2), 1337},
            1e-8
        ));
    }
}
