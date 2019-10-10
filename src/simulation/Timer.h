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

/**
 * @brief A convenient timer for performance checks.
 */
class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;

    time_point startTime;
    time_point finishTime;

public:
    /**
     * @brief Saves the time point of the start.
     */
    void start();

    /**
     * @brief Saves the time point of the end.
     */
    void stop();


    /**
     * @brief Returns the difference between end and start in microseconds.
     */
    unsigned long countMicroseconds() const;
};

#endif /* TIMER_H_ */
