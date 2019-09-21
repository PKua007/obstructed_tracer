/*
 * MSDData.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef MSDDATA_H_
#define MSDDATA_H_

#include <iosfwd>
#include <vector>

class MSDData {
public:
    struct Entry {
        float x{};
        float y{};
        float x2{};
        float y2{};
        float xy{};

        Entry &operator/=(float divisor);
    };

private:
    std::vector<Entry> data;

public:
    using iterator = std::vector<Entry>::iterator;
    using const_iterator = std::vector<Entry>::const_iterator;

    MSDData(std::size_t numberOfSteps) : data(numberOfSteps + 1) { }

    const Entry &operator[](std::size_t index) const { return this->data.at(index); }
    Entry &operator[](std::size_t index) { return this->data.at(index); }
    std::size_t size() const { return this->data.size(); }
    iterator begin() { return this->data.begin(); }
    iterator end() { return this->data.end(); }
    const_iterator begin() const { return this->data.begin(); }
    const_iterator end() const { return this->data.end(); }

    void store(std::ostream &out) const;
    void restore(std::istream &in);
};

MSDData::Entry operator+(const MSDData::Entry &first, const MSDData::Entry &second);
std::ostream &operator<<(std::ostream &out, MSDData::Entry entry);

#endif /* MSDDATA_H_ */
