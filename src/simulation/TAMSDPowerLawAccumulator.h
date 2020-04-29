/*
 * TAMSDPowerLawAccumulator.h
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#ifndef TAMSDPOWERLAWACCUMULATOR_H_
#define TAMSDPOWERLAWACCUMULATOR_H_

#include "TimeAveragedMSD.h"

class TAMSDPowerLawAccumulator {
private:
    double relativeFitStart{};
    double relativeFitEnd{};

    std::size_t numMSDs{};
    std::vector<double> exponentHistogram;
    TimeAveragedMSD ensembleAveragedTAMSD;
    double averageExponent{};

public:
    TAMSDPowerLawAccumulator(double relativeFitStart, double relativeFitEnd);

    void addTAMSD(const TimeAveragedMSD &tamsd);

    std::vector<double> getExponentHistogram() const;
    TimeAveragedMSD getEnsembleAveragedTAMSD() const;
    double getAverageExponent() const;
};

#endif /* TAMSDPOWERLAWACCUMULATOR_H_ */
