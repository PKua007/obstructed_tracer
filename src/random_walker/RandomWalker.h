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
    std::size_t     numberOfSteps{};
    float           tracerRadius{};
    Move            drift{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<Trajectory> trajectories;

    Trajectory runSingleTrajectory();

public:
    RandomWalker(std::size_t numberOfWalks, std::size_t numberOfSteps, float tracerRadius, Move drift,
                 MoveGenerator *moveGenerator, MoveFilter *moveFilter);

    void run(std::ostream &logger);
    std::size_t getNumberOfTrajectories() const;
    const Trajectory &getTrajectory(std::size_t index) const;
};

#endif /* RANDOMWALKER_H_ */
