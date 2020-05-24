/*
 * TimeAveragedMSD.h
 *
 *  Created on: 28 kwi 2020
 *      Author: pkua
 */

#ifndef TIMEAVERAGEDMSD_H_
#define TIMEAVERAGEDMSD_H_

#include <vector>
#include <iosfwd>

/**
 * @brief Mean square displacement averaged over the trajectory.
 * @details It is a function of Delta and calculated as
 * \f$ 1/(T-\Delta) \int_0^{T-\Delta} (\vec{r}(t+\Delta) - \vec{r}(t))^2 dt\f$,
 * where T is a whole trajectory time.
 */
class TimeAveragedMSD {
public:
    struct Entry {
        float delta2{};
        float delta{};

        Entry() { }
        Entry(float delta2, float delta) : delta2{delta2}, delta{delta} { }

        friend Entry operator+(const Entry &e1, const Entry &e2);
        friend Entry operator/(const Entry &tamsd, float a);
        friend std::ostream &operator<<(std::ostream &out, const Entry &entry);
    };

private:
    std::vector<Entry> data;
    std::size_t stepSize;
    float integrationStep;

public:
    using iterator = std::vector<Entry>::iterator;
    using const_iterator = std::vector<Entry>::const_iterator;

    TimeAveragedMSD() { }

    /**
     * @brief Created TA MSD with @a numSteps steps, with stepSize @a stepSize, where these are expressed in number of
     * iterations.
     * @details Note, that the first step corresponds to Delta=0, and last to Delta=@a stepSize*(@a numSteps - 1).
     * @a integrationStep is used to translate everything to real times.
     */
    TimeAveragedMSD(std::size_t numSteps, std::size_t stepSize, float integrationStep);

    Entry &operator[](std::size_t stepIdx);
    Entry operator[](std::size_t stepIdx) const;
    float getVariance(std::size_t stepIdx) const;

    std::size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

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

    /**
     * @brief Stores TA MSD - each step is a separate text line of format [real time] [value]
     */
    void store(std::ostream &out) const;

    TimeAveragedMSD &operator+=(const TimeAveragedMSD &other);
    friend TimeAveragedMSD operator/(const TimeAveragedMSD &tamsd, float a);
};

#endif /* TIMEAVERAGEDMSD_H_ */
