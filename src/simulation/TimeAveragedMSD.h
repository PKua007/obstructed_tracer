/*
 * TimeAveragedMSD.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSD_H_
#define TIMEAVERAGEDMSD_H_

#include <vector>

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
     * @a integrationStep is used to translate everything to real times.
     */
    TimeAveragedMSD(std::size_t numSteps, std::size_t stepSize, float integrationStep);

    float &operator[](std::size_t stepIdx);
    float operator[](std::size_t stepIdx) const;

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

    /**
     * @brief Fits power law to the relative range of data given by @a relativeFitStart and @a relativeFitEnd and
     * returns the exponent.
     */
    double getPowerLawExponent(double relativeFitStart, double relativeFitEnd) const;

    TimeAveragedMSD &operator+=(const TimeAveragedMSD &other);
    friend TimeAveragedMSD operator/(const TimeAveragedMSD &tamsd, float a);
};

#endif /* TIMEAVERAGEDMSD_H_ */
