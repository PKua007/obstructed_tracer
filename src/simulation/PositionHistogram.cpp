/*
 * PositionHistogram.cpp
 *
 *  Created on: 1 kwi 2020
 *      Author: pkua
 */

#include <algorithm>
#include <iterator>

#include "PositionHistogram.h"
#include "utils/Assertions.h"

PositionHistogram::PositionHistogram(std::vector<std::size_t> timeSteps) : timeSteps{std::move(timeSteps)} {
    this->histograms.resize(this->timeSteps.size());
}

void PositionHistogram::addTrajectories(const RandomWalker &walker) {
    std::size_t numberOfTrajectories = walker.getNumberOfTrajectories();
    for (std::size_t trajIdx{}; trajIdx < numberOfTrajectories; trajIdx++) {
        const Trajectory &trajectory = walker.getTrajectory(trajIdx);
        Point initialPosition = trajectory[0];
        for (std::size_t stepIdx{}; stepIdx < this->timeSteps.size(); stepIdx++) {
            std::size_t step = this->timeSteps[stepIdx];
            Assert(step < trajectory.getSize());
            this->histograms[stepIdx].push_back(trajectory[step] - initialPosition);
        }
    }
}

void PositionHistogram::printForStep(std::size_t step, std::ostream &out) {
    auto stepIt = std::find(this->timeSteps.begin(), this->timeSteps.end(), step);
    Expects(stepIt != this->timeSteps.end());
    std::size_t stepIdx = stepIt - this->timeSteps.begin();

    const auto &histogram = this->histograms[stepIdx];
    std::copy(histogram.begin(), histogram.end(), std::ostream_iterator<Move>(out, "\n"));
}
