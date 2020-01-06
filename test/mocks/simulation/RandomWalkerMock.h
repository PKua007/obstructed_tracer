/*
 * RandomWalkerMock.h
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#include <catch2/trompeloeil.hpp>

#include "simulation/RandomWalker.h"


class RandomWalkerMock : public RandomWalker
{
public:
    MAKE_MOCK0(getRandomInitialTracersVector, std::vector<Tracer>(), override);
    MAKE_MOCK2(run, void(std::ostream &, const std::vector<Tracer> &), override);
    MAKE_CONST_MOCK0(getNumberOfTrajectories, std::size_t(), override);
    MAKE_CONST_MOCK0(getNumberOfSteps, std::size_t(), override);
    MAKE_CONST_MOCK1(getTrajectory, const Trajectory &(std::size_t), override);
};
