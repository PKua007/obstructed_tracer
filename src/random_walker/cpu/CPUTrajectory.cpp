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
    if (this->getSize() != 0)
        Expects(this->getLast() == trajectory.getFirst());

    // If this is empty, the initial tracer should be included
    for (std::size_t i = (this->getSize() == 0 ? 0 : 1); i < trajectory.getSize(); i++)
        this->trajectory.push_back(trajectory[i]);
    this->acceptedSteps += trajectory.getNumberOfAcceptedSteps();
}
