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
#include "Tracer.h"

class Trajectory {
private:
    std::vector<Point> data;
    std::size_t acceptedSteps{};

public:
    Trajectory() = default;
    Trajectory(std::size_t numberOfPoints);

    void stayStill();
    void moveToPoint(Tracer tracer);

    std::size_t getSize() const;
    std::size_t getNumberOfAcceptedSteps() const;
    Point operator[](std::size_t index) const;
    Point getFirst() const;
    Point getLast() const;

    void store(std::ostream &out) const;
};

#endif /* TRAJECTORY_H_ */
