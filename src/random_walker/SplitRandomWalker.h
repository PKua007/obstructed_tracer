/*
 * SplitRandomWalker.h
 *
 *  Created on: 18 paź 2019
 *      Author: pkua
 */

#ifndef SPLITRANDOMWALKER_H_
#define SPLITRANDOMWALKER_H_

#include <vector>

#include "simulation/RandomWalker.h"
#include "cpu/CPUTrajectory.h"

/**
 * @brief RandomWalker, which encapsulated another RandomWalker, and performs walks in parts.
 *
 * If we imagine that trajectories are rows of a 2d array, it used one underlying RandomWalker run to fill first n
 * columns (so n first steps of all trajectories), then second run to next n columns and so on. The initial positions
 * for the first run are taken from underlying RandomWalker RandomWalker::randomInitialTracersVector method.
 */
class SplitRandomWalker : public RandomWalker {
private:
    std::size_t numberOfStepsPerSplit{};
    std::size_t numberOfSplits{};
    std::size_t numberOfTrajectories{};
    RandomWalker &randomWalker;
    std::vector<CPUTrajectory> trajectories;

public:
    SplitRandomWalker(std::size_t numberOfSplits, RandomWalker &randomWalker);

    std::vector<Tracer> getRandomInitialTracersVector() override;
    void run(std::ostream &logger, const std::vector<Tracer> &initialTracers) override;
    std::size_t getNumberOfTrajectories() const override;
    std::size_t getNumberOfSteps() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* SPLITRANDOMWALKER_H_ */
