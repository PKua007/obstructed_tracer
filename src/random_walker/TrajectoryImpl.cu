/*
 * TrajectoryImpl.cpp
 *
 *  Created on: 20 paź 2019
 *      Author: pkua
 */

#include <iterator>

#include "TrajectoryImpl.h"
#include "utils/Assertions.h"
#include "utils/CudaCheck.h"


TrajectoryImpl::TrajectoryImpl(std::size_t numberOfPoints) {
    // + 1 for the initial tracer
    this->trajectory.reserve(numberOfPoints + 1);
}

std::size_t TrajectoryImpl::getSize() const {
    return this->trajectory.size();
}

std::size_t TrajectoryImpl::getNumberOfAcceptedSteps() const {
    return this->acceptedSteps;
}

Point TrajectoryImpl::operator[](std::size_t index) const {
    return this->trajectory[index];
}

Point TrajectoryImpl::getFirst() const {
    return this->trajectory.front();
}

Point TrajectoryImpl::getLast() const{
    return this->trajectory.back();
}

void TrajectoryImpl::clear() {
    this->trajectory.clear();
    this->acceptedSteps = 0;
}

void TrajectoryImpl::store(std::ostream &out) const {
    std::copy(this->trajectory.begin(), this->trajectory.end(), std::ostream_iterator<Point>(out, "\n"));
}

void TrajectoryImpl::addPoint(Point point, bool isAccepted) {
    Expects ( !(this->getSize() == 0 && isAccepted) );

    this->trajectory.push_back(point);
    if (isAccepted)
        this->acceptedSteps++;
}

void TrajectoryImpl::appendAnotherTrajectory(const Trajectory &trajectory) {
    if (this->getSize() != 0)
        Expects(this->getLast() == trajectory.getFirst());

    // If this is empty, the initial tracer should be included
    for (std::size_t i = (this->getSize() == 0 ? 0 : 1); i < trajectory.getSize(); i++)
        this->trajectory.push_back(trajectory[i]);
    this->acceptedSteps += trajectory.getNumberOfAcceptedSteps();
}

void TrajectoryImpl::copyGPUData(Point* gpuData, std::size_t size, std::size_t acceptedSteps) {
    this->acceptedSteps = acceptedSteps;
    this->trajectory.resize(size);
    cudaCheck( cudaMemcpy(this->trajectory.data(), gpuData, size*sizeof(Point), cudaMemcpyDeviceToHost) );
}
