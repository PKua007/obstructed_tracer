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

#include "simulation/RandomWalker.h"
#include "TrajectoryImpl.h"
#include "MoveGenerator.h"
#include "MoveFilter.h"

/**
 * @brief CPU implementation of RandomWalker performing walks in parallel using OpenMP.
 *
 * The number of walks is specified in the constructor and the procedure is such as described in RandomWalker. The
 * behaviour of move generation and accepting can be steered by MoveGenerator and MoveFilter classes. The memory
 * for trajectories is allocated on the construction and old trajectories are overwritten every run.
 */
class CPURandomWalker : public RandomWalker {
private:
    std::size_t     numberOfTrajectories{};
    std::size_t     numberOfSteps{};
    float           tracerRadius{};
    Move            drift{};
    MoveGenerator   *moveGenerator{};
    MoveFilter      *moveFilter{};
    std::vector<TrajectoryImpl> trajectories;

    TrajectoryImpl runSingleTrajectory(Tracer initialTracer);

public:
    /**
     * @brief Constructs the walker.
     *
     * It allocates the memory for @a numberOfWalks with parameters @a walkParameters. The additional parameters,
     * @a moveGenerator and @a moveFilter are strategies of sampling moves and accepting them. The MoveFilter will be
     * prepared using MoveFilter::setupFilterForTracer radius with radius taken from @a walkParameters.
     *
     * @param numberOfWalks number of walks to be performed in parallel using OpenMP loop
     * @param walkParameters the parameter of each random walk
     * @param moveGenerator strategy of sampling moves
     * @param moveFilter strategy of accepting moves and choosing starting positions of the tracer
     * @param logger the output stream on which info about preparing @a moveFilter will be printed
     */
    CPURandomWalker(std::size_t numberOfWalks, WalkParameters walkParameters, MoveGenerator *moveGenerator,
                    MoveFilter *moveFilter, std::ostream &logger);

    /**
     * @brief Performs random walks which number was set in the constructor in parallel using OpenMP.
     *
     * @param logger the output stream which will be used to print some info about the progress
     * @param initialTracers initial tracer positions for random walks
     */
    void run(std::ostream &logger, const std::vector<Tracer> &initialTracers) override;

    std::vector<Tracer> getRandomInitialTracersVector() override;
    std::size_t getNumberOfTrajectories() const override;
    std::size_t getNumberOfSteps() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* CPURANDOMWALKER_H_ */
