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
public:
    TAMSDPowerLawAccumulator(double relativeFitStart, double relativeFitEnd);

    void addTAMSD(const TimeAveragedMSD &tamsd);

    std::vector<double> getExponentHistogram() const;
    TimeAveragedMSD getEnsembleAveragedTAMSD() const;
    double getAverageExponent() const;
};

#endif /* TAMSDPOWERLAWACCUMULATOR_H_ */
