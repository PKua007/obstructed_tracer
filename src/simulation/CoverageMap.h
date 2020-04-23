/*
 * CoverageMap.h
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#ifndef COVERAGEMAP_H_
#define COVERAGEMAP_H_

#include "IntegerPoint.h"

class CoverageMap {
private:
    std::size_t width;
    std::size_t height;
    std::vector<std::size_t> coverageMap;

    std::size_t integetPointToIndex(IntegerPoint integerPoint);

public:
    CoverageMap(std::size_t width, std::size_t height);

    void visit(IntegerPoint integerPoint);
    bool isVisited(IntegerPoint integerPoint);
    std::size_t numOfVisits(IntegerPoint integerPoint);

    friend CoverageMap operator+(const CoverageMap &map1, const CoverageMap &map2);
};

#endif /* COVERAGEMAP_H_ */
