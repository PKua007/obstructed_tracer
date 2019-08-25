/*
 * CPUTrajectory.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include "CPUTrajectory.h"

#include <ostream>

CPUTrajectory::CPUTrajectory(std::size_t numberOfPoints) {
    this->data.reserve(numberOfPoints);
}

Point CPUTrajectory::getFirst() const {
    return this->data.front();
}

Point CPUTrajectory::getLast() const {
    return this->data.back();
}

std::size_t CPUTrajectory::getNumberOfAcceptedSteps() const {
    return this->acceptedSteps;
}

void CPUTrajectory::stayStill() {
    this->data.push_back(this->data.back());
}

void CPUTrajectory::store(std::ostream& out) const {
    for (auto point : this->data)
        out << point.x << " " << point.y << "\n";
}

void CPUTrajectory::moveToPoint(Tracer tracer) {
    this->data.push_back(tracer.getPosition());
    this->acceptedSteps++;
}

std::size_t CPUTrajectory::getSize() const {
    return this->data.size();
}

Point CPUTrajectory::operator[](std::size_t index) const {
    return this->data[index];
}
