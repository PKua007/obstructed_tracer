/*
 * TAMSDErgodicityBreakingAccumulator.h
 *
 *  Created on: 18 wrz 2020
 *      Author: pkua
 */

#ifndef TAMSDERGODICITYBREAKINGACCUMULATOR_H_
#define TAMSDERGODICITYBREAKINGACCUMULATOR_H_

#include <vector>

#include "TimeAveragedMSD.h"

/**
 * @brief A clas taking and ensemble of TA MSDs and calculating ergodicity breaking dependendence on Delta from them.
 * @detail EB is an ensemble variance on TA MSD (so 4-th degree in length unit) divided by ensemble averaged TA MSD
 * squared to make it unitless.
 */
class TAMSDErgodicityBreakingAccumulator {
private:
    std::size_t numSteps{};
    std::size_t stepSize{};
    float integrationStep{};

    std::size_t numMSDs{};

    std::vector<double> delta2;
    std::vector<double> delta2Squared;

public:
    /**
     * @brief Parameters here should match those from incoming TA MSD-s.
     */
    TAMSDErgodicityBreakingAccumulator(std::size_t numSteps, std::size_t stepSize, float integrationStep);

    void addTAMSD(const TimeAveragedMSD &tamsd);

    /**
     * @brief Gives EB values for all steps matching those from constructor/incoming TA MSD-s.
     */
    std::vector<double> getEBParameters() const;

    /**
     * @brief Stores results from getEBParameters, row by row, with format: [real time] [EB]
     */
    void storeEBParameters(std::ostream &out) const;
};

#endif /* TAMSDERGODICITYBREAKINGACCUMULATOR_H_ */
