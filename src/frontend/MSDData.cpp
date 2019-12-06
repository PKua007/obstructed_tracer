/*
 * MSDData.cpp
 *
 *  Created on: 21 wrz 2019
 *      Author: pkua
 */

#include <ostream>
#include <istream>
#include <algorithm>
#include <iterator>

#include "MSDData.h"
#include "utils/Utils.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"


MSDData::Entry operator+(const MSDData::Entry &first, const MSDData::Entry &second) {
    MSDData::Entry result;
    result.x = first.x + second.x;
    result.y = first.y + second.y;
    result.x2 = first.x2 + second.x2;
    result.y2 = first.y2 + second.y2;
    result.xy = first.xy + second.xy;
    return result;
}

MSDData::Entry &MSDData::Entry::operator/=(float divisor) {
    this->x /= divisor;
    this->y /= divisor;
    this->x2 /= divisor;
    this->y2 /= divisor;
    this->xy /= divisor;
    return *this;
}

std::ostream &operator<<(std::ostream &out, MSDData::Entry entry) {
    out << entry.x << " " << entry.y << " " << entry.x2 << " " << entry.y2 << " " << entry.xy;
    return out;
}

std::istream &operator>>(std::istream &in, MSDData::Entry &entry) {
    std::istream::sentry sEntry(in);
    if (!sEntry)
        return in;

    in >> entry.x >> entry.y >> entry.x2 >> entry.y2 >> entry.xy;
    return in;
}

void MSDData::store(std::ostream &out) const {
    std::copy(this->data.begin(), this->data.end(), std::ostream_iterator<Entry>(out, "\n"));
}

void MSDData::restore(std::istream &out) {
    this->data.clear();
    std::copy(std::istream_iterator<Entry>(out), std::istream_iterator<Entry>(), std::back_inserter(this->data));
}
