/*
 * RandomWalker.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>
#include <iostream>

#include "RandomWalker.h"

RandomWalker::RandomWalker(std::size_t numberOfSteps, MoveGenerator *moveGenerator, MoveFilter *moveFilter) :
        numberOfSteps{numberOfSteps}, moveGenerator{moveGenerator}, moveFilter{moveFilter} {

}

Trajectory RandomWalker::run() {
    Trajectory trajectory(this->numberOfSteps + 1);

    Point tracer = this->moveFilter->randomValidPoint();
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
