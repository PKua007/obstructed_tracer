/*
 * SimulationTimer.h
 *
 *  Created on: 9 wrz 2019
 *      Author: pkua
 */

#ifndef SIMULATIONTIMER_H_
#define SIMULATIONTIMER_H_

#include <chrono>
#include <iosfwd>

class SimulationTimer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;

    std::size_t numberOfTrajectories;
    time_point startTime;
    time_point finishTime;

public:
    SimulationTimer(std::size_t numberOfTrajectories) : numberOfTrajectories{numberOfTrajectories} { }

    void start();
    void stop();
    void showInfo(std::ostream &logger) const;
};

#endif /* SIMULATIONTIMER_H_ */
