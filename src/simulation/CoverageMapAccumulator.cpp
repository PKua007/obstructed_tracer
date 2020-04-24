/*
 * CoverageMapAccumulator.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include "CoverageMapAccumulator.h"
#include "utils/OMPDefines.h"

CoverageMapAccumulator::CoverageMapAccumulator(std::size_t width, std::size_t height)
        : width{width}, height{height}, map(width, height), mapCountedOnce(width, height)
{

}

void CoverageMapAccumulator::addTrajectories(const std::vector<Trajectory> &trajectories) {
    std::vector<CoverageMap> threadMaps;
    std::vector<CoverageMap> threadMapsCountedOnce;

    for (std::size_t i{}; i < _OMP_MAXTHREADS; i++) {
        threadMaps.push_back(CoverageMap(this->width, this->height));
        threadMapsCountedOnce.push_back(CoverageMap(this->width, this->height));
    }

    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < trajectories.size(); i++) {
        auto &trajectory = trajectories[i];

        CoverageMap trajectoryMapCountedOnce(this->width, this->height);
        for (Point point : trajectory) {
            threadMaps[_OMP_THREAD_ID].visit(point);
            if (!trajectoryMapCountedOnce.isVisited(point))
                trajectoryMapCountedOnce.visit(point);
        }
        threadMapsCountedOnce[_OMP_THREAD_ID] += trajectoryMapCountedOnce;
    }

    for (const auto &map : threadMaps)
        this->map += map;
    for (const auto &map : threadMapsCountedOnce)
        this->mapCountedOnce += map;
}

CoverageMap CoverageMapAccumulator::getCoverageMapCountedOnce() const {
    return this->mapCountedOnce;
}

CoverageMap CoverageMapAccumulator::getCoverageMap() const {
    return this->map;
}
