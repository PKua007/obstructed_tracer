/*
 * MSDData.h
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#ifndef MSDDATA_H_
#define MSDDATA_H_

#include <iosfwd>
#include <vector>

#include "RandomWalker.h"

class MSDData {
private:
    struct Entry {
        float x{};
        float y{};
        float x2{};
        float y2{};
        float xy{};
    };

    friend Entry operator+(const Entry &first, const Entry &second);
    friend std::ostream &operator<<(std::ostream &out, Entry msdData);

    std::size_t numberOfSteps{};
    std::size_t numberOfTrajectories{};
    std::vector<Entry> data;

public:
    MSDData(std::size_t numberOfSteps) : numberOfSteps{numberOfSteps}, data(numberOfSteps + 1) { }
    void addTrajectories(const RandomWalker &randomWalker);
    void store(std::ostream &out);
};

std::ostream &operator<<(std::ostream &out, MSDData::Entry entry);

#endif /* MSDDATA_H_ */
