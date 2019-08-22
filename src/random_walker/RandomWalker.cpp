/*
 * RandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>

#include "RandomWalker.h"
#include "../utils/Assertions.h"

RandomWalker::RandomWalker(std::size_t numberOfSteps, float tracerRadius, Move drift, MoveGenerator *moveGenerator,
                           MoveFilter *moveFilter)
        : numberOfSteps{numberOfSteps}, tracerRadius{tracerRadius}, drift{drift}, moveGenerator{moveGenerator},
          moveFilter{moveFilter} {
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);
}

Trajectory RandomWalker::run() {
    Trajectory trajectory(this->numberOfSteps + 1);

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
