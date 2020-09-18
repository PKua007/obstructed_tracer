/*
 * TAMSDErgodicityBreakingAccumulator.cpp
 *
 *  Created on: 18 wrz 2020
 *      Author: pkua
 */

#include "TAMSDErgodicityBreakingAccumulator.h"
#include "utils/Assertions.h"

TAMSDErgodicityBreakingAccumulator::TAMSDErgodicityBreakingAccumulator(std::size_t numSteps, std::size_t stepSize,
                                                                       float integrationStep)
        : delta2(numSteps), delta2Squared(numSteps), stepSize{stepSize}, integrationStep{integrationStep}
{
    Expects(numSteps > 0);
    Expects(stepSize > 0);
    Expects(integrationStep > 0);
}

void TAMSDErgodicityBreakingAccumulator::addTAMSD(const TimeAveragedMSD &tamsd) {
    Expects(tamsd.getNumSteps() == this->delta2.size());
    Expects(tamsd.getStepSize() == this->stepSize);
    Expects(tamsd.getIntegrationStep() == this->integrationStep);

    for (std::size_t i{}; i < this->delta2.size(); i++) {
        delta2[i] += tamsd[i].delta2;
        delta2Squared[i] += std::pow(tamsd[i].delta2, 2);
    }

    this->numMSDs++;
}

std::vector<double> TAMSDErgodicityBreakingAccumulator::getEBParameters() const {
    Assert(this->numMSDs > 0);
    std::vector<double> EBs(this->delta2.size());
    for (std::size_t i{}; i < this->delta2.size(); i++) {
        if (this->delta2[i] == 0)
            EBs[i] = 0;
        else
            EBs[i] = (this->delta2Squared[i] * this->numMSDs / std::pow(this->delta2[i], 2)) - 1;
    }
    return EBs;
}

void TAMSDErgodicityBreakingAccumulator::storeEBParameters(std::ostream &out) const {
    auto EBs = this->getEBParameters();
    for (std::size_t i{}; i < EBs.size(); i++) {
        double EB = EBs[i];
        double t = this->integrationStep * this->stepSize * i;
        out << t << " " << EB << std::endl;
    }
}
