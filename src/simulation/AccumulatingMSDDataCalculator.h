/*
 * AccumulatingMSDDataCalculator.h
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#ifndef ACCUMULATINGMSDDATACALCULATOR_H_
#define ACCUMULATINGMSDDATACALCULATOR_H_

#include <iosfwd>
#include <vector>

#include "frontend/MSDData.h"
#include "RandomWalker.h"

/**
 * @brief A class, which can "consume" some trajectories and compute MSDData for all of them.
 *
 * Trajectories can be added one after another, so they do not have to all exist in the memory at the same time.
 * After fetching, the class resets its state.
 */
class AccumulatingMSDDataCalculator {
private:
    std::size_t numberOfTrajectories{};
    MSDData data;

public:
    /**
     * @brief Creates the calculator for trajectories with number of steps determined by the first trajectory
     */
    AccumulatingMSDDataCalculator() { }

    /**
     * @brief Creates the calculator for trajectories with @a numberOfSteps + 1 steps (+1 for the initial tracer
     * position).
     * @param numberOfSteps number of steps in the trajectory not including the initial tracer position
     */
    AccumulatingMSDDataCalculator(std::size_t numberOfSteps) : data(numberOfSteps) { }

    /**
     * @brief Adds a trajectory fetched from @a randomWalker to the mean.
     *
     * The operations are performed in parellel using OpenMP.
     *
     * @param randomWalker random walker to fetch the trajectory from
     */
    void addTrajectories(const RandomWalker &randomWalker);

    /**
     * @brief Returns the MSDData obtained from previously added trajectories using addTrajectories and clears the
     * data from the class.
     * @return calculated MSDData
     */
    MSDData fetchMSDData();
};


#endif /* ACCUMULATINGMSDDATACALCULATOR_H_ */
