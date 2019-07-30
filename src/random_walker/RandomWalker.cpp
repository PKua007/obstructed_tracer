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

RandomWalker::RandomWalker(std::size_t numberOfSteps, float tracerRadius, MoveGenerator *moveGenerator,
                           MoveFilter *moveFilter)
        : numberOfSteps{numberOfSteps}, tracerRadius{tracerRadius}, moveGenerator{moveGenerator},
          moveFilter{moveFilter} {
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);
}

Trajectory RandomWalker::run() {
    Trajectory trajectory(this->numberOfSteps + 1);

    Point initialPosition = this->moveFilter->randomValidPoint();
    Tracer tracer(initialPosition, this->tracerRadius);
    trajectory.addPoint(tracer);

    for (std::size_t i = 0; i < this->numberOfSteps; i++) {
        Move move = this->moveGenerator->generateMove();
        if (this->moveFilter->isMoveValid(tracer, move)) {
            tracer += move;
            trajectory.addPoint(tracer);
        }
    }

    return trajectory;
}
