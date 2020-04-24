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
#include "simulation/Trajectory.h"
#include "simulation/MoveGenerator.h"
#include "simulation/MoveFilter.h"
#include "simulation/Timer.h"

/**
 * @brief A GPU implementation of RandomWalker using CUDA.
 *
 * The trajectories are computed in parallel using GPU kernel. The number of walks and parameters of them are set in the
 * constructors. It also accepts pointers to MoveFilter and MoveGenerator allocated on the GPU which determined how
 * moves are generated and filtered. The class manages the memory on GPU used for trajectories.
 */
class GPURandomWalker : public RandomWalker {
private:
    /* Helper class managing the memory for trajectories */
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

        /* The constructor which reserves gpu memory for numberfOfTrajectories trajectrories of numberOfSteps steps
         * (excluding initial tracer position). */
        TrajectoriesOnGPU(std::size_t numberOfTrajectories, std::size_t numberOfSteps);

        ~TrajectoriesOnGPU();
        TrajectoriesOnGPU(const TrajectoriesOnGPU &) = delete;
        TrajectoriesOnGPU &operator=(const TrajectoriesOnGPU &) = delete;

        /* Methods which provide arrays allocated on GPU to be used on GPU */
        Point **getTrajectoriesArray() { return this->gpuArrayOfGPUTrajectories; }
        size_t *getAcceptedStepsArray() { return this->gpuArrayOfAcceptedSteps; }

        /* Copies trajectory data from GPU to CPU. The vector has to be initialized to have size equal to
         * numberOfTrajectories, however the trajectories in the vector themselves require no special treatment.
         */
        void copyToCPU(std::vector<Trajectory> &trajectories);
    };

    std::size_t     numberOfTrajectories{};
    WalkParameters  walkParameters;
    std::size_t     numberOfMoveFilterSetupThreads{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<Trajectory> trajectories;
    TrajectoriesOnGPU trajectoriesOnGPU;

    static constexpr int blockSize = 512;

    void setupMoveFilterForTracerRadius(std::ostream& logger);
    void printTimerInfo(const Timer &kernelTimer, const Timer &copyTimer, std::ostream &logger);

public:
    /**
     * @brief Construct the random walker based on the parameters.
     *
     * This constructor accepts MoveGenerator and MoveFilter strategies to customize how moves are generated and
     * filtered. This classes have to be allocated on the GPU. It setups ImageMoveFilter for tracer radius given in
     * @a walkParameters using @a numberOfMoveFilterSetupThreads GPU threads (see MoveFilter::setupForTracerRadius).
     * Some info is spit to @a logger. Also, the GPU memory is allocated and persists during the life of an object.
     *
     * @param numberOfWalks number of walks to be performed in parallel
     * @param walkParameters the parameters of all walks
     * @param numberOfMoveFilterSetupThreads the number of threads which will be used to setup @a moveFilter for
     * tracer radius
     * @param moveGenerator GPU-allocated MoveGenerator sampling random moves
     * @param moveFilter GPU-allocated MoveFilter for accpeting moves and sampling random initial tracers
     * @param logger the output stream to show info on MoveFilter setup
     */
    GPURandomWalker(std::size_t numberOfWalks, WalkParameters walkParameters,
                    std::size_t numberOfMoveFilterSetupThreads, MoveGenerator *moveGenerator, MoveFilter *moveFilter,
                    std::ostream &logger);

    GPURandomWalker(const GPURandomWalker &) = delete;
    GPURandomWalker &operator=(const GPURandomWalker &) = delete;
    ~GPURandomWalker();

    /**
     * @brief Performs parallel GPU walks based on parameters given in the constructor.
     *
     * After the walks results on GPU are copied to CPU and can be accessed.
     *
     * @param logger output stream to show progress
     * @param initialTracers initial tracer positions for random walks
     */
    void run(std::ostream &logger, const std::vector<Tracer> &initialTracers) override;

    std::vector<Tracer> getRandomInitialTracersVector() override;
    std::size_t getNumberOfSteps() const override;
    std::size_t getNumberOfTrajectories() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
    const std::vector<Trajectory> &getTrajectories() const override;
};

#endif /* GPURANDOMWALKER_H_ */
