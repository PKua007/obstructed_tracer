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

class PositionHistogram {
private:
    std::vector<std::size_t> timeSteps;
    std::vector<std::vector<Point>> histograms;

public:
    PositionHistogram() { }
    explicit PositionHistogram(std::vector<std::size_t> timeSteps);

    void addTrajectories(const RandomWalker &walker);
    void printForStep(std::size_t step, std::ostream &out);
};

#endif /* POSITIONHISTOGRAM_H_ */
