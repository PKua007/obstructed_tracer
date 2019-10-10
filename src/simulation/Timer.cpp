/*
 * Timer.cpp
 *
 *  Created on: 9 wrz 2019
 *      Author: pkua
 */

#include <ostream>

#include "Timer.h"

void Timer::start() {
    this->startTime = clock::now();
}

void Timer::stop() {
    this->finishTime = clock::now();
}

unsigned long Timer::countMicroseconds() const {
    using namespace std::chrono;

    auto timeDifference = this->finishTime - this->startTime;
    return duration_cast<microseconds>(timeDifference).count();
}
