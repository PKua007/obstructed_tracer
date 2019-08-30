/*
 * CPUTrajectory.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include "CPUTrajectory.h"

#include <ostream>

CPUTrajectory::CPUTrajectory(std::size_t numberOfPoints) {
    this->trajectory.reserve(numberOfPoints);
}

void CPUTrajectory::stayStill() {
    this->trajectory.push_back(this->trajectory.back());
}

void CPUTrajectory::moveToPoint(Point point) {
    this->trajectory.push_back(point);
    this->acceptedSteps++;
}
