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
#include "../utils/Assertions.h"

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
    Tracer tracer = this->moveFilter->randomValidTracer(this->tracerRadius);
    trajectory.moveToPoint(tracer);
    for (std::size_t i = 0; i < this->numberOfSteps; i++) {
        Move move = this->moveGenerator->generateMove() + drift;
        if (this->moveFilter->isMoveValid(tracer, move)) {
            tracer += move;
            trajectory.moveToPoint(tracer);
        } else {
            trajectory.stayStill();
        }
    }
    return trajectory;
}

void CPURandomWalker::run(std::ostream &logger) {
    for (std::size_t i = 0; i < this->trajectories.size(); i++) {
        logger << "[RandomWalker::run] Starting walk " << i << "... " << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        this->trajectories[i] = this->runSingleTrajectory();
        auto finish = std::chrono::high_resolution_clock::now();

        auto mus = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        logger << " Finished after " << mus << " microseconds. Initial position: " << this->trajectories[i].getFirst();
        logger << ", accepted steps: " << this->trajectories[i].getNumberOfAcceptedSteps() << ", final position: ";
        logger << this->trajectories[i].getLast() << std::endl;
    }
}

std::size_t CPURandomWalker::getNumberOfTrajectories() const {
    return this->trajectories.size();
}

const Trajectory& CPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
