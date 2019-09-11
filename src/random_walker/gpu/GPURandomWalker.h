/*
 * GPURandomWalker.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPURANDOMWALKER_H_
#define GPURANDOMWALKER_H_

#include <vector>
#include "simulation/RandomWalker.h"
#include "GPUTrajectory.h"
#include "../MoveGenerator.h"
#include "../MoveFilter.h"

class GPURandomWalker : public RandomWalker {
private:
    class TrajectoriesOnGPU {
    private:
        std::size_t numberOfTrajectories{};
        std::size_t numberOfSteps{};

        // We need GPU pointers to trajectories in both CPU and GPU array
        Point **gpuArrayOfGPUTrajectories{};
        std::vector<Point*> cpuVectorOfGPUTrajectories;

        // We need number of accepted steps in both CPU and GPU arrays
        size_t *gpuArrayOfAcceptedSteps{};
        std::vector<std::size_t> cpuVectorOfAcceptedSteps;

    public:

        TrajectoriesOnGPU(std::size_t numberOfTrajectories, std::size_t numberOfSteps);
        ~TrajectoriesOnGPU();
        TrajectoriesOnGPU(TrajectoriesOnGPU &other) = delete;
        TrajectoriesOnGPU &operator=(TrajectoriesOnGPU &other) = delete;

        Point **getTrajectoriesArray() { return this->gpuArrayOfGPUTrajectories; }
        size_t *getAcceptedStepsArray() { return this->gpuArrayOfAcceptedSteps; }

        void copyToCPU(std::vector<GPUTrajectory> &trajectories);
    };

    std::size_t     numberOfTrajectories{};
    WalkParameters  walkParameters;
    std::size_t     numberOfMoveFilterSetupThreads{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<GPUTrajectory> trajectories;
    TrajectoriesOnGPU trajectoriesOnGPU;

    static constexpr int blockSize = 512;

    void setupMoveFilterForTracerRadius(std::ostream& logger);

public:
    GPURandomWalker(std::size_t numberOfWalks, WalkParameters walkParameters,
                    std::size_t numberOfMoveFilterSetupThreads, MoveGenerator *moveGenerator, MoveFilter *moveFilter,
                    std::ostream &logger);

    void run(std::ostream &logger) override;
    std::size_t getNumberOfTrajectories() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* GPURANDOMWALKER_H_ */
