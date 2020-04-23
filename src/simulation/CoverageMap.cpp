/*
 * CoverageMap.cpp
 *
 *  Created on: 23 kwi 2020
 *      Author: pkua
 */

#include "CoverageMap.h"

CoverageMap::CoverageMap(std::size_t width, std::size_t height) {

}

void CoverageMap::visit(IntegerPoint integerPoint) {

}

bool CoverageMap::isVisited(IntegerPoint integerPoint) {
    return false;
}

CoverageMap operator+(const CoverageMap &map1, const CoverageMap &map2) {
    return CoverageMap(0, 0);
}

std::size_t CoverageMap::numOfVisits(IntegerPoint integerPoint) {
    return 0;
}
