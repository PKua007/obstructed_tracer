/*
 * SplitRandomWalker.cpp
 *
 *  Created on: 18 paź 2019
 *      Author: pkua
 */

#include <algorithm>
#include <functional>

#include "SplitRandomWalker.h"
#include "simulation/Timer.h"
#include "utils/Assertions.h"

SplitRandomWalker::SplitRandomWalker(std::size_t numberOfSplits, std::unique_ptr<RandomWalker> randomWalker)
        : randomWalker{std::move(randomWalker)}, numberOfTrajectories{randomWalker->getNumberOfTrajectories()},
          numberOfStepsPerSplit{randomWalker->getNumberOfSteps()}, numberOfSplits{numberOfSplits}
{
    Expects(this->numberOfSplits > 0);
    this->trajectories.resize(this->numberOfTrajectories);
    for (auto &trajectory : this->trajectories)
        trajectory = Trajectory(this->numberOfSplits * this->numberOfStepsPerSplit);
}

std::vector<Tracer> SplitRandomWalker::getRandomInitialTracersVector() {
    return this->randomWalker->getRandomInitialTracersVector();
}

void SplitRandomWalker::printRangeInfo(std::size_t i, std::ostream &logger) const {
    std::size_t startStep = i * this->numberOfStepsPerSplit;
    std::size_t endStep = startStep + this->numberOfStepsPerSplit - 1;
    std::size_t totalNumberOfSteps = this->numberOfStepsPerSplit * this->numberOfSplits;
    logger << "[SplitRandomWalker::run] Simulating steps " << startStep << " - " << endStep << " out of ";
    logger << totalNumberOfSteps << std::endl;
}

void SplitRandomWalker::printTimerInfo(const Timer &timer, std::ostream &logger) const {
    auto simulationTimeInMus = timer.countMicroseconds();
    auto singleRunTimeInMus = simulationTimeInMus / this->numberOfTrajectories;
    logger << "[SplitRadnomWalker::run] Whole trajectories finished after " << simulationTimeInMus;
    logger << " μs, which gives " << singleRunTimeInMus << " μs per trajectory on average." << std::endl;
}

void SplitRandomWalker::run(std::ostream &logger, const std::vector<Tracer> &initialTracers) {
    for (auto &trajectory : this->trajectories)
        trajectory.clear();

    Timer timer;
    timer.start();
    std::vector<Tracer> currentInitialTracers(initialTracers.begin(), initialTracers.end());
    for (std::size_t i = 0; i < this->numberOfSplits; i++) {
        this->printRangeInfo(i, logger);

        this->randomWalker->run(logger, currentInitialTracers);
        for (std::size_t j = 0; j < this->numberOfTrajectories; j++)
            this->trajectories[j].appendAnotherTrajectory(this->randomWalker->getTrajectory(j));

        if (i < this->numberOfSplits - 1) {
            std::transform(this->trajectories.begin(), this->trajectories.end(), currentInitialTracers.begin(),
                           std::mem_fn(&Trajectory::getLast));
        }
    }
    timer.stop();
    this->printTimerInfo(timer, logger);
}

std::size_t SplitRandomWalker::getNumberOfTrajectories() const {
    return this->trajectories.size();
}

const Trajectory &SplitRandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}

std::size_t SplitRandomWalker::getNumberOfSteps() const {
    return this->numberOfStepsPerSplit * this->numberOfSplits;
}

const std::vector<Trajectory> &SplitRandomWalker::getTrajectories() const {
    return this->trajectories;
}
