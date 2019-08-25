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

MSDData::MSDData(const RandomWalker &randomWalker) {
    std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
    std::size_t trajectorySize = randomWalker.getTrajectory(0).getSize();
    this->data = std::vector<Entry>(trajectorySize);
    for (std::size_t i = 0; i < numberOfTrajectories; i++) {
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

    for (auto& entry : this->data) {
        entry.x2 /= numberOfTrajectories;
        entry.y2 /= numberOfTrajectories;
        entry.xy /= numberOfTrajectories;
    }
}

void MSDData::store(std::ostream &out) const {
    std::copy(this->data.begin(), this->data.end(), std::ostream_iterator<Entry>(out, "\n"));
}

std::ostream &operator<<(std::ostream &out, MSDData::Entry entry) {
    out << entry.x2 << " " << entry.y2 << " " << entry.xy;
}
