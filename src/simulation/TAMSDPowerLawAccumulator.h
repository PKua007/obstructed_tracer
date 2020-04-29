/*
 * TAMSDPowerLawAccumulator.h
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#ifndef TAMSDPOWERLAWACCUMULATOR_H_
#define TAMSDPOWERLAWACCUMULATOR_H_

#include "TimeAveragedMSD.h"

/**
 * @brief A class which calculates and collects some information about TA MSD from a several TA MSD.
 * @details There are: histogram of exponents of power law, ensemble averaged TA MSD and and exponent for it.
 */
class TAMSDPowerLawAccumulator {
private:
    double relativeFitStart{};
    double relativeFitEnd{};

    std::size_t numMSDs{};
    std::vector<double> exponentHistogram;
    TimeAveragedMSD ensembleAveragedTAMSD;
    double averageExponent{};

public:
    /**
     * @brief @a relativeFitStart and @a relativeFitEnd will tell to what part of data a power law should be fitted.
     */
    TAMSDPowerLawAccumulator(double relativeFitStart, double relativeFitEnd);

    void addTAMSD(const TimeAveragedMSD &tamsd);

    std::vector<double> getExponentHistogram() const;
    TimeAveragedMSD getEnsembleAveragedTAMSD() const;
    double getEnsembleAverageExponent() const;
};

#endif /* TAMSDPOWERLAWACCUMULATOR_H_ */
