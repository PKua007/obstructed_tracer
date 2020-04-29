/*
 * TAMSDPowerLawAccumulator.cpp
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#include "TAMSDPowerLawAccumulator.h"

TAMSDPowerLawAccumulator::TAMSDPowerLawAccumulator(double relativeFitStart, double relativeFitEnd) {

}

void TAMSDPowerLawAccumulator::addTAMSD(const TimeAveragedMSD &tamsd) {

}

std::vector<double> TAMSDPowerLawAccumulator::getExponentHistogram() const {
    return {};
}

TimeAveragedMSD TAMSDPowerLawAccumulator::getEnsembleAveragedTAMSD() const {
    return TimeAveragedMSD{};
}

double TAMSDPowerLawAccumulator::getAverageExponent() const {
    return 0.0;
}
