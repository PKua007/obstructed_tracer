/*
 * GPURandomWalker.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <stdexcept>

#include "GPURandomWalker.h"

void GPURandomWalker::run(std::ostream& logger) {

}

std::size_t GPURandomWalker::getNumberOfTrajectories() const {
    return 0;
}

const Trajectory& GPURandomWalker::getTrajectory(std::size_t index) const {
    throw std::runtime_error("stub");
}
