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
#include "simulation/Timer.h"

CPURandomWalker::CPURandomWalker(std::size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                                 MoveGenerator *moveGenerator, MoveFilter *moveFilter, std::ostream &logger)
        : numberOfTrajectories{numberOfTrajectories}, numberOfSteps{walkParameters.numberOfSteps},
          tracerRadius{walkParameters.tracerRadius}, drift{walkParameters.drift}, moveGenerator{moveGenerator},
          moveFilter{moveFilter}
{
    Expects(this->numberOfTrajectories > 0);
    Expects(this->numberOfSteps > 0);
    Expects(this->tracerRadius >= 0.f);
    this->trajectories.resize(numberOfTrajectories);

    logger << "[CPURandomWalker] Preparing MoveFilter... " << std::flush;
    this->moveFilter->setupForTracerRadius(this->tracerRadius);
    logger << "done." << std::endl;
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
    logger << "[CPURandomWalker::run] Simulating: " << std::flush;

    Timer timer;
    timer.start();
    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        this->trajectories[i] = this->runSingleTrajectory();

        if (i % 100 == 99) {
            _OMP_CRITICAL(stdout)
            logger << "." << std::flush;
        }
    }
    timer.stop();
    logger << " completed." << std::endl;

    auto simulationTimeInMus = timer.count();
    auto singleRunTimeInMus = simulationTimeInMus / this->numberOfTrajectories;
    logger << "[CPURandomWalker::run] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;
}

std::size_t CPURandomWalker::getNumberOfTrajectories() const {
    return this->numberOfTrajectories;
}

const Trajectory& CPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
