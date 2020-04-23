/*
 * CoverageMap.h
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#ifndef COVERAGEMAP_H_
#define COVERAGEMAP_H_

#include "IntegerPoint.h"

/**
 * @brief A simple class counting visits at 2d map (with pbc)
 */
class CoverageMap {
private:
    std::size_t width;
    std::size_t height;
    std::vector<std::size_t> coverageMap;

    std::size_t integerPointToIndex(IntegerPoint integerPoint) const;

public:
    CoverageMap(std::size_t width, std::size_t height);

    /**
     * @brief Increases the counter for a given point (w.r.t. pbc)
     */
    void visit(IntegerPoint integerPoint);
    bool isVisited(IntegerPoint integerPoint) const;
    std::size_t numOfVisits(IntegerPoint integerPoint) const;

    /**
     * @brief Creates a new map from corresponding entries in 2 given maps.
     */
    friend CoverageMap operator+(const CoverageMap &map1, const CoverageMap &map2);
};

#endif /* COVERAGEMAP_H_ */
