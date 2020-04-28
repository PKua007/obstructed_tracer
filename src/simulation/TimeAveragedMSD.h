/*
 * TimeAveragedMSD.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSD_H_
#define TIMEAVERAGEDMSD_H_

#include <vector>

#include "utils/Assertions.h"

class TimeAveragedMSD {
private:
    std::vector<float> data;
    std::size_t stepSize;

public:
    using iterator = std::vector<float>::iterator;
    using const_iterator = std::vector<float>::const_iterator;

    TimeAveragedMSD() { }
    TimeAveragedMSD(std::size_t deltaSteps, std::size_t stepSize) : stepSize{stepSize}, data(deltaSteps) { }

    float &operator[](std::size_t stepIdx) {
        Expects(stepIdx < this->size());
        return data[stepIdx];
    }

    float operator[](std::size_t stepIdx) const {
        Expects(stepIdx < this->size());
        return data[stepIdx];
    }

    std::size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator begin() const { data.begin(); }
    const_iterator end() const { data.end(); }
    std::size_t getStepSize() { return stepSize; }
};

#endif /* TIMEAVERAGEDMSD_H_ */
