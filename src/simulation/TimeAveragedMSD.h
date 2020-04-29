/*
 * TimeAveragedMSD.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSD_H_
#define TIMEAVERAGEDMSD_H_

#include <vector>
#include <algorithm>

#include "utils/Assertions.h"
#include "analyzer/PowerRegression.h"

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
    float integrationStep;

public:
    using iterator = std::vector<float>::iterator;
    using const_iterator = std::vector<float>::const_iterator;

    TimeAveragedMSD() { }

    /**
     * @brief Created TA MSD with @a numSteps steps, with stepSize @a stepSize, where these are expressed in number of
     * iterations.
     * @details Note, that the first step corresponds to Delta=0, and last to Delta=@a stepSize*(@a numSteps - 1).
     */
    TimeAveragedMSD(std::size_t numSteps, std::size_t stepSize, float integrationStep)
            : stepSize{stepSize}, data(numSteps), integrationStep(integrationStep)
    { }

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

    /**
     * @brief Based on @a stepSize and @a integrationStep passed in the constructor, translates data index to the
     * physical time.
     */
    float dataIndexToRealTime(std::size_t index) const { return index * this->stepSize * this->integrationStep; }

    double getPowerLawExponent(double relativeFitStart, double relativeFitEnd) const {
        Expects(relativeFitStart > 0);
        Expects(relativeFitEnd > relativeFitStart);
        Expects(relativeFitEnd <= 1);

        std::size_t fitStartIdx = static_cast<std::size_t>(this->size()*relativeFitStart);
        std::size_t fitEndIdx = static_cast<std::size_t>(this->size()*relativeFitEnd);
        Assert(fitStartIdx > 0);
        Assert(fitEndIdx < this->size());

        PowerRegression regression;
        for (std::size_t i = fitStartIdx; i < fitEndIdx; i++)
            regression.addXY(this->dataIndexToRealTime(i), this->data[i]);
        regression.calculate();
        return regression.getExponent().value;
    }

    TimeAveragedMSD &operator+=(const TimeAveragedMSD &other) {
        Expects(this->size() == other.size());
        Expects(this->stepSize == other.stepSize);
        Expects(this->integrationStep == other.integrationStep);

        std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(),
                       std::plus<float>());
        return *this;
    }

    friend TimeAveragedMSD operator/(const TimeAveragedMSD &tamsd, float a) {
        TimeAveragedMSD result(tamsd.size(), tamsd.stepSize, tamsd.integrationStep);
        for (std::size_t i{}; i < tamsd.size(); i++)
            result[i] = tamsd.data[i] / a;
        return result;
    }
};

#endif /* TIMEAVERAGEDMSD_H_ */
