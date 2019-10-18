/*
 * CPUTrajectory.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include "CPUTrajectory.h"
#include "utils/Assertions.h"

#include <ostream>

CPUTrajectory::CPUTrajectory(std::size_t numberOfPoints, Point initialPosition) {
    // + 1 for initial tracer
    this->trajectory.reserve(numberOfPoints + 1);
    this->trajectory.push_back(initialPosition);
}

void CPUTrajectory::stayStill() {
    this->trajectory.push_back(this->trajectory.back());
}

void CPUTrajectory::moveToPoint(Point point) {
    this->trajectory.push_back(point);
    this->acceptedSteps++;
}

void CPUTrajectory::appendAnotherTrajectory(const Trajectory &trajectory) {
    Expects(this->getLast() == trajectory.getFirst());

    for (std::size_t i = 1; i < trajectory.getNumberOfAcceptedSteps(); i++)
        this->trajectory.push_back(trajectory[i]);
    this->acceptedSteps += trajectory.getNumberOfAcceptedSteps();
}
