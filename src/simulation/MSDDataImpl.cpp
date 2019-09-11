/*
 * MSDData.cpp
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <iterator>
#include <algorithm>

#include "MSDDataImpl.h"
#include "utils/Utils.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"


MSDDataImpl::Entry operator+(const MSDDataImpl::Entry &first, const MSDDataImpl::Entry &second) {
    return {first.x + second.x, first.y + second.y, first.x2 + second.x2, first.y2 + second.y2, first.xy + second.xy};
}

std::ostream &operator<<(std::ostream &out, MSDDataImpl::Entry entry) {
    out << entry.x << " " << entry.y << " " << entry.x2 << " " << entry.y2 << " " << entry.xy;
    return out;
}

void MSDDataImpl::store(std::ostream &out) {
    for (auto &entry : this->data) {
        entry.x /= this->numberOfTrajectories;
        entry.y /= this->numberOfTrajectories;
        entry.x2 /= this->numberOfTrajectories;
        entry.y2 /= this->numberOfTrajectories;
        entry.xy /= this->numberOfTrajectories;
    }

    std::copy(this->data.begin(), this->data.end(), std::ostream_iterator<Entry>(out, "\n"));
}

void MSDDataImpl::addTrajectories(const RandomWalker &randomWalker) {
    std::size_t numberOfNewTrajectories = randomWalker.getNumberOfTrajectories();
    Assert(numberOfNewTrajectories > 0);

    std::size_t trajectorySize = randomWalker.getTrajectory(0).getSize();
    Assert(trajectorySize > 0);

    std::vector<std::vector<Entry>> threadPartialSums(_OMP_MAXTHREADS);
    for (auto &partialSum : threadPartialSums)
        partialSum.resize(trajectorySize);

    _OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < numberOfNewTrajectories; i++) {
        std::size_t threadId = _OMP_THREAD_ID;
        auto &trajectory = randomWalker.getTrajectory(i);

        float startX = trajectory[0].x;
        float startY = trajectory[0].y;
        for (std::size_t j = 0; j < trajectorySize; j++) {
            float x = trajectory[j].x - startX;
            float y = trajectory[j].y - startY;

            threadPartialSums[threadId][j].x += x;
            threadPartialSums[threadId][j].y += y;
            threadPartialSums[threadId][j].x2 += x*x;
            threadPartialSums[threadId][j].y2 += y*y;
            threadPartialSums[threadId][j].xy += x*y;
        }
    }

    for (const auto &partialSum : threadPartialSums)
        std::transform(this->data.begin(), this->data.end(), partialSum.begin(), this->data.begin(), std::plus<Entry>());

    this->numberOfTrajectories += numberOfNewTrajectories;
}
