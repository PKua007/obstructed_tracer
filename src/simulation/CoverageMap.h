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
public:
    CoverageMap(std::size_t width, std::size_t height);

    void visit(IntegerPoint integerPoint);
    bool isVisited(IntegerPoint integerPoint);
    std::size_t numOfVisits(IntegerPoint integerPoint);
};

CoverageMap operator+(const CoverageMap &map1, const CoverageMap &map2);

#endif /* COVERAGEMAP_H_ */
