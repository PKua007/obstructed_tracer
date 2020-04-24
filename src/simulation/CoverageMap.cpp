/*
 * CoverageMap.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include <algorithm>

#include "CoverageMap.h"
#include "utils/Assertions.h"

namespace {
    int mod(int a, int b) { return (a % b + b) % b; }
}

CoverageMap::CoverageMap(std::size_t width, std::size_t height)
        : width{width}, height{height}, coverageMap(width * height)
{
    Expects(width > 0);
    Expects(height > 0);
}

void CoverageMap::visit(IntegerPoint integerPoint) {
    this->coverageMap[this->integerPointToIndex(integerPoint)]++;
}

bool CoverageMap::isVisited(IntegerPoint integerPoint) const {
    return this->coverageMap[this->integerPointToIndex(integerPoint)] > 0;
}

std::size_t CoverageMap::numOfVisits(IntegerPoint integerPoint) const {
    return this->coverageMap[this->integerPointToIndex(integerPoint)];
}

std::size_t CoverageMap::integerPointToIndex(IntegerPoint integerPoint) const {
    return mod(integerPoint.x, this->width) + mod(integerPoint.y, this->height) * this->width;
}

CoverageMap &CoverageMap::operator+=(const CoverageMap &other) {
    (*this) = (*this) + other;
    return (*this);
}

CoverageMap operator+(const CoverageMap &map1, const CoverageMap &map2) {
    Expects(map1.width == map2.width);
    Expects(map1.height == map2.height);

    CoverageMap map(map1.width, map2.height);
    std::transform(map1.coverageMap.begin(), map1.coverageMap.end(), map2.coverageMap.begin(), map.coverageMap.begin(),
                   std::plus<std::size_t>());
    return map;
}
