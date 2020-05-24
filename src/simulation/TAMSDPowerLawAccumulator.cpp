/*
 * TAMSDPowerLawAccumulator.cpp
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#include "TAMSDPowerLawAccumulator.h"

#include "utils/Assertions.h"

TAMSDPowerLawAccumulator::TAMSDPowerLawAccumulator(double relativeFitStart, double relativeFitEnd)
        : relativeFitStart{relativeFitStart}, relativeFitEnd{relativeFitEnd}
{
    Expects(relativeFitStart > 0);
    Expects(relativeFitEnd > relativeFitStart);
    Expects(relativeFitEnd <= 1);
}

void TAMSDPowerLawAccumulator::addTAMSD(const TimeAveragedMSD &tamsd) {
    if (this->ensembleAveragedTAMSD.empty())
        this->ensembleAveragedTAMSD = tamsd;
    else
        this->ensembleAveragedTAMSD += tamsd;    // This also checks if the new TA MDS has the same params as the rest

    this->numMSDs++;
    this->exponentHistogram.push_back(tamsd.getPowerLawExponent(this->relativeFitStart, this->relativeFitEnd));
    this->varianceExponentHistogram.push_back(tamsd.getVariancePowerLawExponent(this->relativeFitStart, this->relativeFitEnd));
}

std::vector<double> TAMSDPowerLawAccumulator::getExponentHistogram() const {
    return this->exponentHistogram;
}

std::vector<double> TAMSDPowerLawAccumulator::getVarianceExponentHistogram() const {
    return this->varianceExponentHistogram;
}

TimeAveragedMSD TAMSDPowerLawAccumulator::getEnsembleAveragedTAMSD() const {
    return this->ensembleAveragedTAMSD / this->numMSDs;
}

double TAMSDPowerLawAccumulator::getEnsembleAveragedExponent() const {
    return this->getEnsembleAveragedTAMSD().getPowerLawExponent(this->relativeFitStart, this->relativeFitEnd);
}

double TAMSDPowerLawAccumulator::getEnsembleAveragedVarianceExponent() const {
    return this->getEnsembleAveragedTAMSD().getVariancePowerLawExponent(this->relativeFitStart, this->relativeFitEnd);
}
