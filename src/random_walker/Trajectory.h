/*
 * Trajectory.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include <vector>

#include "Point.h"

class Trajectory {
private:
    std::vector<Point> data;

public:
    Trajectory() = default;
    Trajectory(std::size_t numberOfPoints);

    void addPoint(Point point);

    std::size_t getSize() const;
    Point operator[](std::size_t index) const;
    Point getFirst() const;
    Point getLast() const;

    void store(std::ostream &out) const;
};

#endif /* TRAJECTORY_H_ */
