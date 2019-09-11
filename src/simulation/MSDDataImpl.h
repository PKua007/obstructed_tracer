/*
 * MSDDataImpl.h
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#ifndef MSDDATAIMPL_H_
#define MSDDATAIMPL_H_

#include <iosfwd>
#include <vector>

#include "MSDData.h"
#include "RandomWalker.h"

class MSDDataImpl : public MSDData {
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
    MSDDataImpl(std::size_t numberOfSteps) : numberOfSteps{numberOfSteps}, data(numberOfSteps + 1) { }

    void store(std::ostream &out) override;

    void addTrajectories(const RandomWalker &randomWalker);
};

MSDDataImpl::Entry operator+(const MSDDataImpl::Entry &first, const MSDDataImpl::Entry &second);
std::ostream &operator<<(std::ostream &out, MSDDataImpl::Entry entry);

#endif /* MSDDATAIMPL_H_ */
