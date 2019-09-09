/*
 * GPURandomWalker.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPURANDOMWALKER_H_
#define GPURANDOMWALKER_H_

#include <vector>
#include "RandomWalker.h"
#include "GPUTrajectory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"

class GPURandomWalker : public RandomWalker {
private:
    std::size_t     numberOfTrajectories{};
    std::size_t     numberOfSteps{};
    std::size_t     numberOfMoveFilterSetupThreads{};
    float           tracerRadius{};
    Move            drift{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<GPUTrajectory> trajectories;

    static constexpr int blockSize = 512;

    void setupMoveFilterForTracerRadius(std::ostream& logger);

public:
    GPURandomWalker(std::size_t numberOfWalks, std::size_t numberOfSteps, std::size_t numberOfMoveFilterSetupThreads,
                    float tracerRadius, Move drift, MoveGenerator *moveGenerator, MoveFilter *moveFilter);

    void run(std::ostream &logger) override;
    std::size_t getNumberOfTrajectories() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* GPURANDOMWALKER_H_ */
