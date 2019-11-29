/*
 * CPURandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>
#include <algorithm>
#include <functional>

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

TrajectoryImpl CPURandomWalker::runSingleTrajectory(Tracer initialTracer) {
    Tracer tracer = initialTracer;
    TrajectoryImpl trajectory(this->numberOfSteps);
    trajectory.addPoint(tracer.getPosition());

    for (std::size_t i = 0; i < this->numberOfSteps; i++) {
        Move move = this->moveGenerator->generateMove() + drift;
        if (this->moveFilter->isMoveValid(tracer, move)) {
            tracer += move;
            trajectory.addPoint(tracer.getPosition(), true);
        } else {
            trajectory.addPoint(tracer.getPosition(), false);
        }
    }
    return trajectory;
}

void CPURandomWalker::run(std::ostream &logger, const std::vector<Tracer> &initialTracers) {
    logger << "[CPURandomWalker::run] Simulating: " << std::flush;

    Timer timer;
    timer.start();
    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        this->trajectories[i] = this->runSingleTrajectory(initialTracers[i]);

        if (i % 100 == 99) {
            _OMP_CRITICAL(stdout)
            logger << "." << std::flush;
        }
    }
    timer.stop();
    logger << " completed." << std::endl;

    auto simulationTimeInMus = timer.countMicroseconds();
    auto singleRunTimeInMus = simulationTimeInMus / this->numberOfTrajectories;
    logger << "[CPURandomWalker::run] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;
}

std::size_t CPURandomWalker::getNumberOfTrajectories() const {
    return this->numberOfTrajectories;
}

std::size_t CPURandomWalker::getNumberOfSteps() const {
    return this->numberOfSteps;
}

std::vector<Tracer> CPURandomWalker::getRandomInitialTracersVector() {
    std::vector<Tracer> result(this->numberOfTrajectories);
    std::generate(result.begin(), result.end(), std::bind(&MoveFilter::randomValidTracer, this->moveFilter));
    return result;
}

const Trajectory &CPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
