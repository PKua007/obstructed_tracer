/*
 * Timer.h
 *
 *  Created on: 9 wrz 2019
 *      Author: pkua
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <iosfwd>

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;

    time_point startTime;
    time_point finishTime;

public:
    void start();
    void stop();
    unsigned long count() const;
};

#endif /* SIMULATIONTIMER_H_ */
