/*
 * CPURandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>

#include "CPURandomWalker.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"
#include "simulation/SimulationTimer.h"

CPURandomWalker::CPURandomWalker(std::size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                                 MoveGenerator *moveGenerator, MoveFilter *moveFilter)
        : numberOfTrajectories{numberOfTrajectories}, numberOfSteps{walkParameters.numberOfSteps},
          tracerRadius{walkParameters.tracerRadius}, drift{walkParameters.drift}, moveGenerator{moveGenerator},
          moveFilter{moveFilter}
{
    Expects(this->numberOfTrajectories > 0);
    Expects(this->numberOfSteps > 0);
    Expects(this->tracerRadius >= 0.f);
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

    SimulationTimer timer(this->numberOfTrajectories);
    timer.start();
    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        this->trajectories[i] = this->runSingleTrajectory();

        _OMP_CRITICAL(stdout)
        logger << "." << std::flush;
    }
    timer.stop();
    logger << " completed." << std::endl;

    timer.showInfo(logger);
}

std::size_t CPURandomWalker::getNumberOfTrajectories() const {
    return this->numberOfTrajectories;
}

const Trajectory& CPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
