/*
 * CPUTrajectory.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>

#include "CPUTrajectory.h"
#include "utils/Assertions.h"

CPUTrajectory::CPUTrajectory(std::size_t numberOfPoints) {
    // + 1 for the initial tracer
    this->trajectory.reserve(numberOfPoints + 1);
}

void CPUTrajectory::addPoint(Point point, bool isAccepted) {
    Expects ( !(this->getSize() == 0 && isAccepted) );

    this->trajectory.push_back(point);
    if (isAccepted)
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
