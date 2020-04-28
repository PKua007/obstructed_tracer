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

class TimeAveragedMSDCalculator {
public:
    TimeAveragedMSDCalculator(std::size_t maxDelta, std::size_t deltaStep);

    TimeAveragedMSD calculate(const Trajectory &trajectory);
};

#endif /* TIMEAVERAGEDMSDCALCULATOR_H_ */
