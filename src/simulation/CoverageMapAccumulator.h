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
