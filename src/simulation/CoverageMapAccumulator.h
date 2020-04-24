/*
 * CoverageMapAccumulator.h
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#ifndef COVERAGEMAPACCUMULATOR_H_
#define COVERAGEMAPACCUMULATOR_H_

#include <vector>

#include "Trajectory.h"
#include "CoverageMap.h"

/**
 * @brief A class accumulating coverage maps in 2 ways for each given trajectory, w.r.t. pbc.
 * @detail One way is calculating each point (getCoverageMap()) normally, and the second one is taking point repeating
 * in a single trajectory as only one and then add to the map (getCoverageMapCountedOnce())
 */
class CoverageMapAccumulator {
private:
    std::size_t width;
    std::size_t height;
    CoverageMap map;
    CoverageMap mapCountedOnce;

public:
    CoverageMapAccumulator(std::size_t width, std::size_t height);

    void addTrajectories(const std::vector<Trajectory> &trajectories);
    CoverageMap getCoverageMapCountedOnce() const;
    CoverageMap getCoverageMap() const;
};

#endif /* COVERAGEMAPACCUMULATOR_H_ */
