/*
 * CoverageMapAccumulator.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include "CoverageMapAccumulator.h"

CoverageMapAccumulator::CoverageMapAccumulator(std::size_t width, std::size_t height)
        : width{width}, height{height}, map(width, height), mapCountedOnce(width, height)
{

}

void CoverageMapAccumulator::addTrajectories(const std::vector<Trajectory> &trajectories) {
    for (const auto &trajectory : trajectories) {
        CoverageMap trajectoryMapCountedOnce(this->width, this->height);
        for (Point point : trajectory) {
            this->map.visit(point);
            if (!trajectoryMapCountedOnce.isVisited(point))
                trajectoryMapCountedOnce.visit(point);
        }
        this->mapCountedOnce += trajectoryMapCountedOnce;
    }
}

CoverageMap CoverageMapAccumulator::getCoverageMapCountedOnce() const {
    return this->mapCountedOnce;
}

CoverageMap CoverageMapAccumulator::getCoverageMap() const {
    return this->map;
}
