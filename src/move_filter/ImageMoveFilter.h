/*
 * ImageMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGEMOVEFILTER_H_
#define IMAGEMOVEFILTER_H_

#include <random>

#include "../random_walker/MoveFilter.h"
#include "../image/Image.h"

class ImageMoveFilter: public MoveFilter {
private:
    struct ImageMove {
        int x{};
        int y{};

        ImageMove() = default;
        ImageMove(int x, int y) : x{x}, y{y} { };
    };

    struct ImagePoint {
        int x{};
        int y{};

        ImagePoint() = default;
        ImagePoint(int x, int y) : x{x}, y{y} { };
        ImagePoint(Point point) : x{static_cast<int>(point.x)}, y{static_cast<int>(point.y)} { };

        bool operator==(ImagePoint second) const;
        bool operator!=(ImagePoint second) const;
        ImageMove operator-(ImagePoint second) const;
    };

    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    std::size_t width;
    std::size_t height;
    std::vector<bool> validPointsMap;

    float radiusForTracerCache = -1.0;
    std::vector<std::size_t> validTracerIndicesCache;

    void rebuildValidTracersCache(float radius);

    bool isPointValid(ImagePoint point, float pointRadius) const;
    bool checkValidPointsMap(ImagePoint point) const;
    bool isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const;

    ImagePoint indexToPoint(std::size_t index) const;

public:
    ImageMoveFilter(Image image, unsigned int seed);

    bool isMoveValid(Tracer tracer, Move move) const override;
    Tracer randomValidTracer(float radius) override;
};

#endif /* IMAGEMOVEFILTER_H_ */