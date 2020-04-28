/*
 * TimeAveragedMSD.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSD_H_
#define TIMEAVERAGEDMSD_H_

#include <vector>

class TimeAveragedMSD {
public:
    using iterator = std::vector<float>::iterator;
    using const_iterator = std::vector<float>::const_iterator;

    TimeAveragedMSD() { }
    TimeAveragedMSD(std::size_t deltaSteps, std::size_t stepSize);

    float operator[](std::size_t stepIdx) const;
    std::size_t size() const { return 0; }
    bool empty() const { return false; }
    iterator begin() { return iterator{}; }
    iterator end() { return iterator{}; }
    const_iterator begin() const { return const_iterator{}; }
    const_iterator end() const { return const_iterator{}; }
    std::size_t getDeltaSteps() { return 0; }
    std::size_t getStepSize() { return 0; }
};

#endif /* TIMEAVERAGEDMSD_H_ */
