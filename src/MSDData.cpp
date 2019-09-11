/*
 * MSDData.cpp
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <iterator>

#include "MSDData.h"
#include "utils/Utils.h"


void MSDData::store(std::ostream &out) {
    for (auto &entry : this->data) {
        entry.x2 /= this->numberOfTrajectories;
        entry.y2 /= this->numberOfTrajectories;
        entry.xy /= this->numberOfTrajectories;
    }

    std::copy(this->data.begin(), this->data.end(), std::ostream_iterator<Entry>(out, "\n"));
}

std::ostream &operator<<(std::ostream &out, MSDData::Entry entry) {
    out << entry.x2 << " " << entry.y2 << " " << entry.xy;
    return out;
}

void MSDData::addTrajectories(const RandomWalker &randomWalker) {
    std::size_t numberOfNewTrajectories = randomWalker.getNumberOfTrajectories();
    if (numberOfNewTrajectories == 0)
        return;

    std::size_t trajectorySize = randomWalker.getTrajectory(0).getSize();
    if (trajectorySize == 0)
        return;

    for (std::size_t i = 0; i < numberOfNewTrajectories; i++) {
        auto &trajectory = randomWalker.getTrajectory(i);

        float startX = trajectory[0].x;
        float startY = trajectory[0].y;
        for (std::size_t j = 0; j < trajectory.getSize(); j++) {
            float x = trajectory[j].x - startX;
            float y = trajectory[j].y - startY;
            this->data[j].x2 += x*x;
            this->data[j].y2 += y*y;
            this->data[j].xy += x*y;
        }
    }

    this->numberOfTrajectories += numberOfNewTrajectories;
}
