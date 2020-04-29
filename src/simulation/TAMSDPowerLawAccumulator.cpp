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
    double alpha = tamsd.getPowerLawExponent(this->relativeFitStart, this->relativeFitEnd);
    this->exponentHistogram.push_back(alpha);
    this->averageExponent += alpha;
}

std::vector<double> TAMSDPowerLawAccumulator::getExponentHistogram() const {
    return this->exponentHistogram;
}

TimeAveragedMSD TAMSDPowerLawAccumulator::getEnsembleAveragedTAMSD() const {
    return this->ensembleAveragedTAMSD / this->numMSDs;
}

double TAMSDPowerLawAccumulator::getEnsembleAverageExponent() const {
    return this->averageExponent / this->numMSDs;
}
