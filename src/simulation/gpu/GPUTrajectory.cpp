/*
 * GPUTrajectory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include "GPUTrajectory.h"

std::size_t GPUTrajectory::getSize() const {
    return 0;
}

std::size_t GPUTrajectory::getNumberOfAcceptedSteps() const {
    return 0;
}

Point GPUTrajectory::operator[](std::size_t index) const {
    return {};
}

Point GPUTrajectory::getFirst() const {
    return {};
}

Point GPUTrajectory::getLast() const {
    return {};
}

void GPUTrajectory::store(std::ostream& out) const {

}
