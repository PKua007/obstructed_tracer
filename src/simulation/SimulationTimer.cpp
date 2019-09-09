/*
 * SimulationTimer.cpp
 *
 *  Created on: 9 wrz 2019
 *      Author: pkua
 */

#include <ostream>

#include "SimulationTimer.h"

void SimulationTimer::start() {
    this->startTime = clock::now();
}

void SimulationTimer::stop() {
    this->finishTime = clock::now();
}

void SimulationTimer::showInfo(std::ostream &logger) const {
    using namespace std::chrono;

    auto simulationTime = this->finishTime - this->startTime;
    auto simulationTimeInMus = duration_cast<microseconds>(simulationTime).count();
    auto singleRunTimeInMus = simulationTimeInMus / this->numberOfTrajectories;
    logger << "[SimulationTimer::showInfo] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;
}
