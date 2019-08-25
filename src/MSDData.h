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
        float x2{};
        float y2{};
        float xy{};
    };

    friend std::ostream &operator<<(std::ostream &out, Entry msdData);

    std::vector<Entry> data;

public:
    MSDData(const RandomWalker &randomWalker);
    void store(std::ostream &out) const;
};

std::ostream &operator<<(std::ostream &out, MSDData::Entry entry);

#endif /* MSDDATA_H_ */
