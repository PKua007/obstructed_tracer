/*
 * CPUTrajectory.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef CPUTRAJECTORY_H_
#define CPUTRAJECTORY_H_

#include <vector>

#include "../Trajectory.h"
#include "../Point.h"
#include "Tracer.h"

class CPUTrajectory : public Trajectory {
private:
    std::vector<Point> data;
    std::size_t acceptedSteps{};

public:
    CPUTrajectory() = default;
    CPUTrajectory(std::size_t numberOfPoints);

    void stayStill();
    void moveToPoint(Tracer tracer);

    std::size_t getSize() const override;
    std::size_t getNumberOfAcceptedSteps() const override;
    Point operator[](std::size_t index) const override;
    Point getFirst() const override;
    Point getLast() const override;

    void store(std::ostream &out) const override;
};

#endif /* CPUTRAJECTORY_H_ */
