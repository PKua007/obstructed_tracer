/*
 * CPURandomWalker.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef CPURANDOMWALKER_H_
#define CPURANDOMWALKER_H_

#include <vector>
#include <array>
#include <random>

#include "RandomWalker.h"
#include "CPUTrajectory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"

class CPURandomWalker : public RandomWalker {
private:
    std::size_t     numberOfTrajectories{};
    std::size_t     numberOfSteps{};
    float           tracerRadius{};
    Move            drift{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<CPUTrajectory> trajectories;

    CPUTrajectory runSingleTrajectory();

public:
    CPURandomWalker(std::size_t numberOfWalks, WalkParameters walkParameters, MoveGenerator *moveGenerator,
                    MoveFilter *moveFilter);

    void run(std::ostream &logger) override;
    std::size_t getNumberOfTrajectories() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* CPURANDOMWALKER_H_ */
