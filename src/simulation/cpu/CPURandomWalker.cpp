/*
 * CPURandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>
#include <chrono>

#include "CPURandomWalker.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"

CPURandomWalker::CPURandomWalker(std::size_t numberOfTrajectories, std::size_t numberOfSteps, float tracerRadius,
                                 Move drift, MoveGenerator *moveGenerator, MoveFilter *moveFilter)
        : numberOfSteps{numberOfSteps}, tracerRadius{tracerRadius}, drift{drift}, moveGenerator{moveGenerator},
          moveFilter{moveFilter} {
    Expects(numberOfTrajectories > 0);
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);
    this->trajectories.resize(numberOfTrajectories);
}

CPUTrajectory CPURandomWalker::runSingleTrajectory() {
    CPUTrajectory trajectory(this->numberOfSteps + 1);
    Tracer tracer = this->moveFilter->randomValidTracer();
    trajectory.moveToPoint(tracer.getPosition());
    for (std::size_t i = 0; i < this->numberOfSteps; i++) {
        Move move = this->moveGenerator->generateMove() + drift;
        if (this->moveFilter->isMoveValid(tracer, move)) {
            tracer += move;
            trajectory.moveToPoint(tracer.getPosition());
        } else {
            trajectory.stayStill();
        }
    }
    return trajectory;
}

void CPURandomWalker::run(std::ostream &logger) {
    logger << "[CPURandomWalker::run] Preparing MoveFilter... " << std::flush;
    this->moveFilter->setupForTracerRadius(this->tracerRadius);
    logger << "done." << std::endl;

    logger << "[CPURandomWalker::run] Using up to " << _OMP_MAXTHREADS << " OpenMP threads." << std::endl;
    logger << "[CPURandomWalker::run] Simulating: " << std::flush;

    auto start = std::chrono::high_resolution_clock::now();
    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < this->trajectories.size(); i++) {
        this->trajectories[i] = this->runSingleTrajectory();

        _OMP_CRITICAL(stdout)
        logger << "." << std::flush;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    logger << " completed." << std::endl;

    auto simulationTimeInMus = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    auto singleRunTimeInMus = simulationTimeInMus / this->trajectories.size();
    logger << "[CPURandomWalker::run] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;
}

std::size_t CPURandomWalker::getNumberOfTrajectories() const {
    return this->trajectories.size();
}

const Trajectory& CPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
