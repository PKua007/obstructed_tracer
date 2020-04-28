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

/**
 * @brief Mean square displacement averaged over the trajectory.
 * @details It is a function of Delta and calculated as
 * \f$ 1/(T-\Delta) \int_0^{T-\Delta} (\vec{r}(t+\Delta) - \vec{r}(t))^2 dt\f$,
 * where T is a whole trajectory time.
 */
class TimeAveragedMSD {
private:
    std::vector<float> data;
    std::size_t stepSize;

public:
    using iterator = std::vector<float>::iterator;
    using const_iterator = std::vector<float>::const_iterator;

    TimeAveragedMSD() { }

    /**
     * @brief Created TA MSD with @a numSteps steps, with stepSize @a stepSize, where these are expressed in number of
     * iterations.
     * @details Note, that the first step corresponds to Delta=0, and last to Delta=stepSize*(numSteps - 1)
     */
    TimeAveragedMSD(std::size_t numSteps, std::size_t stepSize) : stepSize{stepSize}, data(numSteps) { }

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
