/*
 * PositionHistogram.h
 *
 *  Created on: 1 kwi 2020
 *      Author: pkua
 */

#ifndef POSITIONHISTOGRAM_H_
#define POSITIONHISTOGRAM_H_

#include <vector>

#include "Point.h"
#include "RandomWalker.h"

/**
 * @brief For each passed RandomWalker, takes all trajectories and saves the positions (relative to start) for specified
 * time points.
 */
class PositionHistogram {
private:
    std::vector<std::size_t> timeSteps;
    std::vector<std::vector<Move>> histograms;

public:
    PositionHistogram() { }

    /**
     * @brief Constructs a histogram builder, which will produce histograms for all time steps specified in
     * @a timeSteps.
     */
    explicit PositionHistogram(std::vector<std::size_t> timeSteps);

    /**
     * @brief Takes all trajectories from @a walker and adds points to the histogram.
     */
    void addTrajectories(const RandomWalker &walker);


    /**
     * @brief Stores all points from histogram for @a step time step in "x y[newline]" format.
     */
    void printForStep(std::size_t step, std::ostream &out);
};

#endif /* POSITIONHISTOGRAM_H_ */
