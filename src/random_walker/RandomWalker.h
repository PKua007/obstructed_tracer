/*
 * RandomWalker.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKER_H_
#define RANDOMWALKER_H_

#include <vector>
#include <array>
#include <random>

#include "Trajectory.h"
#include "MoveGenerator.h"
#include "MoveFilter.h"

class RandomWalker {
private:
    std::size_t numberOfSteps{};
    float tracerRadius{};
    MoveGenerator *moveGenerator{};
    MoveFilter *moveFilter{};

public:
    RandomWalker(std::size_t numberOfSteps, float tracerRadius, MoveGenerator *moveGenerator, MoveFilter *moveFilter);

    Trajectory run();
};

#endif /* RANDOMWALKER_H_ */
