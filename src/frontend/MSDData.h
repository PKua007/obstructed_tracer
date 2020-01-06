/*
 * MSDData.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

/** @file */

#ifndef MSDDATA_H_
#define MSDDATA_H_

#include <iosfwd>
#include <vector>

/**
 * @brief A simple container for mean values of the trajectory.
 *
 * Subsequent elements correspond to mean values of subsequent trajectory steps. See MSDData::Entry to know what data
 * is contained. The class gives some basic methods to access and edit elements, as well as iterators and printing,
 * saving to file and reading from file.
 *
 * @see operator<<(std::ostream &, const MSDData &)
 * @see operator>>(std::istream &, MSDData::Entry &)
 */
class MSDData {
public:
    /**
     * @brief Struct of mean values for a single trajectory step.
     * @see operator+(const MSDData::Entry &, const MSDData::Entry &)
     * @see operator<<(std::ostream &, MSDData::Entry)
     * @see operator>>(std::istream &, MSDData::Entry &)
     */
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

    /**
     * @brief Constructs the empty set.
     */
    MSDData() { }

    /**
     * @brief Constructs the set with @a numberOfSteps + 1 elements (+1 for initial tracer position) with all means set
     * to 0.
     * @param numberOfSteps number of steps in the trajectory not including initial tracer position
     */
    MSDData(std::size_t numberOfSteps) : data(numberOfSteps + 1) { }

    /**
     * @brief Immutable access operator.
     * @param index index of an the element to be accessed
     * @return the element
     */
    const Entry &operator[](std::size_t index) const { return this->data.at(index); }

    /**
     * @brief Mutable access operator.
     * @param index index of an the element to be accessed
     * @return the element
     */
    Entry &operator[](std::size_t index) { return this->data.at(index); }

    /**
     * @brief Returns the size of the set.
     * @return the size of the set
     */
    std::size_t size() const { return this->data.size(); }

    /**
     * @brief Returns the mutable iterator pointing to the first element.
     * @return the mutable iterator pointing to the first element
     */
    iterator begin() { return this->data.begin(); }

    /**
     * @brief Returns the mutable iterator pointing to the element after last.
     * @return the mutable iterator pointing to the first element
     */
    iterator end() { return this->data.end(); }

    /**
     * @brief Returns the immutable iterator pointing to the first element.
     * @return the mutable iterator pointing to the first element
     */
    const_iterator begin() const { return this->data.begin(); }

    /**
     * @brief Returns the immutable iterator pointing to the element after last.
     * @return the mutable iterator pointing to the first element
     */
    const_iterator end() const { return this->data.end(); }

    friend std::ostream &operator<<(std::ostream &, const MSDData &);
    friend std::istream &operator>>(std::istream &, MSDData &);
};

/**
 * @brief Stores the @a msdData in @a out stream.
 *
 * Each line contains one entry, and things in the entry are separated by spaces in the order the same as in
 * MSDData::Entry.
 */
std::ostream &operator<<(std::ostream &out, const MSDData &msdData);

/**
 * @brief Restores the MSD data from @a in stream.
 *
 * Required format: each line contains one entry, and things in the entry are separated by spaces in the order the
 * same as in MSDData::Entry.
 */
std::istream &operator>>(std::istream &in, MSDData &msdData);

/**
 * @brief Sums all fields of two MSDData::Entry objects.
 * @param first lhs operand
 * @param second rhs operand
 * @return new MSDData::Entry obtained by summing all entries of @a first and @a second
 */
MSDData::Entry operator+(const MSDData::Entry &first, const MSDData::Entry &second);

/**
 * @brief Stream insertion operator for MSDData::Entry.
 * @param out stream to print @a entry to
 * @param entry MSDData::Entry to be printed
 * @return reference to @a out
 */
std::ostream &operator<<(std::ostream &out, MSDData::Entry entry);

/**
 * @brief Stream extraction operator for MSDData::Entry.
 * @param in stream to read @a entry from
 * @param entry MSDData::Entry to be read
 * @return reference to @a in
 */
std::istream &operator>>(std::istream &in, MSDData::Entry &entry);

#endif /* MSDDATA_H_ */
