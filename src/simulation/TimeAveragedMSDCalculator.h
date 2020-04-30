/*
 * TimeAveragedMSDCalculator.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSDCALCULATOR_H_
#define TIMEAVERAGEDMSDCALCULATOR_H_

#include "TimeAveragedMSD.h"
#include "Trajectory.h"

/**
 * @brief Calculates TimeAveragedMSD (see that class's description) from a given trajectories.
 */
class TimeAveragedMSDCalculator {
public:
    std::size_t maxDelta;
    std::size_t deltaStep;
    float integrationStep;

public:
    /**
     * @brief Constructs the calculator, which will produce TA MSD, where Delta is from 0 to @a maxDelta with a step
     * @a deltaStep and with integrationStep @a integrationStep.
     */
    TimeAveragedMSDCalculator(std::size_t maxDelta, std::size_t deltaStep, float integrationStep);

    TimeAveragedMSD calculate(const Trajectory &trajectory);

    std::size_t getMaxDelta() const { return this->maxDelta; }
    std::size_t getDeltaStep() const { return this->deltaStep; }
};

#endif /* TIMEAVERAGEDMSDCALCULATOR_H_ */
