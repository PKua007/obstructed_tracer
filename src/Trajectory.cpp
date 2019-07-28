/*
 * Trajectory.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include "Trajectory.h"

#include <ostream>

Trajectory::Trajectory(std::size_t numberOfPoints) {
    this->data.reserve(numberOfPoints);
}

Point Trajectory::getFirst() const {
    return this->data.front();
}

Point Trajectory::getLast() const {
    return this->data.back();
}

void Trajectory::store(std::ostream& out) const {
    for (auto point : this->data)
        out << point.x << " " << point.y << "\n";
}

void Trajectory::addPoint(Point point) {
    this->data.push_back(point);
}

std::size_t Trajectory::getSize() const {
    return this->data.size();
}

Point Trajectory::operator[](std::size_t index) const {
    return this->data[index];
}
