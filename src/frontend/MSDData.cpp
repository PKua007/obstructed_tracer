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

std::ostream &operator<<(std::ostream &out, const MSDData &msdData) {
    std::copy(msdData.data.begin(), msdData.data.end(), std::ostream_iterator<MSDData::Entry>(out, "\n"));
    return out;
}

std::istream &operator>>(std::istream &in, MSDData &msdData) {
    msdData.data.clear();
    std::copy(std::istream_iterator<MSDData::Entry>(in), std::istream_iterator<MSDData::Entry>(),
              std::back_inserter(msdData.data));
    return in;
}

void MSDData::printWithTimes(std::ostream &out, float integrationStep) const {
    std::size_t i{};
    for (const auto &entry : this->data) {
        out << (i * integrationStep) << " " << entry << "\n";
        i++;
    }
}

std::istream &operator>>(std::istream &in, MSDData::TimedEntry &entry) {
    std::istream::sentry sEntry(in);
    if (!sEntry)
        return in;

    in >> entry.t;
    operator>>(in, static_cast<MSDData::Entry &>(entry));
    return in;
}

void MSDData::loadFromFileWithTimes(std::istream &in) {
    this->data.clear();
    std::copy(std::istream_iterator<MSDData::TimedEntry>(in), std::istream_iterator<MSDData::TimedEntry>(),
              std::back_inserter(this->data));
}

