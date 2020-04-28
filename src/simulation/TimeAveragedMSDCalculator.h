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

public:
    /**
     * @brief Constructs the calculator, which will produce TA MSD, where Delta is from 0 to @a maxDelta with a step
     * @a deltaStep.
     */
    TimeAveragedMSDCalculator(std::size_t maxDelta, std::size_t deltaStep);

    TimeAveragedMSD calculate(const Trajectory &trajectory);
};

#endif /* TIMEAVERAGEDMSDCALCULATOR_H_ */
