/*
 * RandomWalker.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKER_H_
#define RANDOMWALKER_H_

#include <vector>

#include "Tracer.h"
#include "Trajectory.h"

/**
 * @brief Interface for class which can perform a series of random walks in parallel.
 *
 * RandomWalker can generate a vector of random inital tracers using underlying MoveFilter. It then is used in a run,
 * where each single walk is perform the following way:
 * <ol>
 *     <li>Use MoveGenerator to generate random move.</li>
 *     <li>Add drift to the move.</li>
 *     <li>Use MoveFilter to check if the move is accepted.</li>
 *     <li>If so, add the new tracer position to the trajectory. If not, copy the last position</li>
 * </ol>
 */
class RandomWalker {
public:
    /**
     * @brief A struct containing walk parameters.
     */
    struct WalkParameters {
        /**
         * @brief Number of steps in the trajectory not including the initial tracer position.
         */
        size_t numberOfSteps;

        /**
         * @brief A radius of the circular tracer.
         */
        float tracerRadius;

        /**
         * @brief A drift to be added to each walk step apart from random move.
         */
        Move drift;
    };

    virtual ~RandomWalker() = default;

    /**
     * @brief Returns a vector of random valid initial tracers using underlying MoveFilter.
     * @return a vector of random valid initial tracers using underlying MoveFilter
     */
    virtual std::vector<Tracer> getRandomInitialTracersVector() = 0;

    /**
     * @brief Performs the multiple random walk of the same type logging some information onto @a logger.
     *
     * The obtained trajectories can be late fetched using getTrajectory.
     *
     * @param logger stream to log information on the process
     * @param initialTracers initial tracer positions for random walks
     */
    virtual void run(std::ostream &logger, const std::vector<Tracer> &initialTracers) = 0;

    /**
     * @brief Return the number of walks performed.
     * @return the number of walks performed
     */
    virtual std::size_t getNumberOfTrajectories() const = 0;

    /**
     * @brief Returns the trajectory of the index @a index.
     * @param index index of the trajectory to fetch
     * @return the tractory of the index @a index
     */
    virtual const Trajectory &getTrajectory(std::size_t index) const = 0;
};

#endif /* RANDOMWALKER_H_ */
