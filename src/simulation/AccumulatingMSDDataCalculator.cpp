/*
 * AccumulatingMSDDataCalculator.cpp
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#include <algorithm>

#include "AccumulatingMSDDataCalculator.h"
#include "utils/Utils.h"
#include "utils/Assertions.h"
#include "utils/OMPDefines.h"


void AccumulatingMSDDataCalculator::addTrajectories(const RandomWalker &randomWalker) {
    std::size_t numberOfNewTrajectories = randomWalker.getNumberOfTrajectories();
    Assert(numberOfNewTrajectories > 0);

    std::size_t trajectorySize = randomWalker.getTrajectory(0).getSize();
    Assert(trajectorySize > 0);

    if (this->data.size() == 0)
        this->data = MSDData(trajectorySize - 1);
    Assert(trajectorySize == this->data.size());

    std::vector<std::vector<MSDData::Entry>> threadPartialSums(_OMP_MAXTHREADS);
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

    for (const auto &partialSum : threadPartialSums) {
        std::transform(this->data.begin(), this->data.end(), partialSum.begin(), this->data.begin(),
                       std::plus<MSDData::Entry>());
    }

    this->numberOfTrajectories += numberOfNewTrajectories;
}

MSDData AccumulatingMSDDataCalculator::fetchMSDData() {
    if (this->numberOfTrajectories == 0)
        return this->data;

    std::size_t dataSize = this->data.size();
    MSDData movedData = std::move(this->data);
    for (auto &entry : movedData)
        entry /= this->numberOfTrajectories;

    this->numberOfTrajectories = 0;
    this->data = MSDData(dataSize - 1);
    return movedData;
}
