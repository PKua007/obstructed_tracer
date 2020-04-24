/*
 * ImageMoveFilter.tpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */


#include "utils/Assertions.h"
#include "utils/Utils.h"

namespace {
    struct IntegerMove {
        int x{};
        int y{};

        CUDA_HOSTDEV IntegerMove(int x, int y) : x{x}, y{y} { };
    };

    CUDA_HOSTDEV IntegerMove operator-(IntegerPoint p1, IntegerPoint p2) {
        return {p1.x - p2.x, p1.y - p2.y};
    }

    template <typename T> CUDA_HOSTDEV int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
}

template <typename BoundaryConditions>
ImageMoveFilter<BoundaryConditions>
    ::ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height, unsigned long seed,
                     size_t numberOfTrajectories)
        : width{width}, height{height}, imageBC(width, height)
{
    this->initializeGenerators(seed, numberOfTrajectories);

    this->validPointsMapSize = this->width * this->height;
    this->validPointsMap = new bool[this->validPointsMapSize];
    this->validTracersMap = new bool[this->validPointsMapSize];

    // On CPU in case of allocation fail std::bad_alloc should be thrown, we only need to check on GPU
    #if CUDA_DEVICE_COMPILATION
        if (this->validPointsMap == nullptr || this->validTracersMap == nullptr) {
            printf("[ImageMoveFilter] Allocation of arrays of valid points failed. Increase the size of GPU heap.\n");
            asm("trap;");
        }
    #endif

    // Image y axis starts from left upper corner downwards, so image is scanned from the bottom left, because
    // validPointsMap is in "normal" coordinate system, with (0, 0) in left bottom corner
    size_t i = 0;
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            IntegerPoint integerPoint = {static_cast<int>(x), static_cast<int>(this->height - y - 1)};
            if (intImageData[this->integerPointToIndex(integerPoint)] == 0xffffffff)
                this->validPointsMap[i] = true;
            else
                this->validPointsMap[i] = false;

            i++;
        }
    }

    // CPU version should be in valid state - for tracer radius 0 - even without calling setupForTracerRadius.
    // GPU version spares some time not doing it
    #if CUDA_HOST_COMPILATION
        this->setupForTracerRadius(0.f);
    #endif
}

template <typename BoundaryConditions>
ImageMoveFilter<BoundaryConditions>
    ::~ImageMoveFilter()
{
    delete [] this->validPointsMap;
    delete [] this->validTracersMap;
    #if CUDA_DEVICE_COMPILATION
        delete [] this->states;
    #endif
}


#if CUDA_DEVICE_COMPILATION

    template <typename BoundaryConditions>
    void
    ImageMoveFilter<BoundaryConditions>
        ::initializeGenerators(unsigned long seed, size_t numberOfTrajectories)
    {
        this->states = new curandState[numberOfTrajectories];
        for (size_t i = 0; i < numberOfTrajectories; i++)
            curand_init(seed, i, 0, &(this->states[i]));
    }

#else // CUDA_HOST_COMPILATION

    template <typename BoundaryConditions>
    void
    ImageMoveFilter<BoundaryConditions>
        ::initializeGenerators(unsigned long seed, size_t numberOfTrajectories)
    {
        this->randomGenerator.seed(seed);
    }

#endif

template <typename BoundaryConditions>
bool
ImageMoveFilter<BoundaryConditions>
    ::isPointValid(IntegerPoint point) const
{
    point = this->imageBC.applyOnIntegerPoint(point);
    return this->validPointsMap[this->integerPointToIndex(point)];
}

template <typename BoundaryConditions>
bool
ImageMoveFilter<BoundaryConditions>
    ::isPrecomputedTracerValid(IntegerPoint position) const
{
    if (!this->imageBC.isIntegerPointInBounds(position, this->tracerRadius))
        return false;

    position = this->imageBC.applyOnIntegerPoint(position);
    return this->validTracersMap[this->integerPointToIndex(position)];
}

template <typename BoundaryConditions>
bool
ImageMoveFilter<BoundaryConditions>
    ::isNotPrecomputedTracerValid(IntegerPoint position, float radius) const
{
    Expects(radius >= 0.f);

    int intPointRadius = static_cast<int>(radius);
    if (!this->imageBC.isIntegerPointInBounds(position, intPointRadius))
        return false;

    if (radius == 0.f)
        return this->isPointValid(position);

    for (int x = -intPointRadius; x <= intPointRadius; x++) {
        for (int y = -intPointRadius; y <= intPointRadius; y++) {
            if (x*x + y*y > radius*radius)
                continue;

            if (!this->isPointValid({position.x + x, position.y + y}))
                return false;
        }
    }
    return true;
}

template <typename BoundaryConditions>
bool
ImageMoveFilter<BoundaryConditions>
    ::isPrecomputedTracerLineValid(IntegerPoint from, IntegerPoint to) const
{
    IntegerMove integerMove = to - from;
    if (abs(integerMove.x) > abs(integerMove.y)) {
        float a = float(integerMove.y) / float(integerMove.x);
        for (int x = from.x; x != to.x; x += sgn(integerMove.x)) {
            int y = static_cast<int>(round(from.y + a * (x - from.x)));
            if (!this->isPrecomputedTracerValid({ x, y }))
                return false;
        }
    } else {
        float a = float(integerMove.x) / float(integerMove.y);
        for (int y = from.y; y != to.y; y += sgn(integerMove.y)) {
            int x = static_cast<int>(round(from.x + a * (y - from.y)));
            if (!this->isPrecomputedTracerValid({ x, y }))
                return false;
        }
    }
    return true;
}

template <typename BoundaryConditions>
IntegerPoint
ImageMoveFilter<BoundaryConditions>
    ::indexToIntegerPoint(size_t index) const
{
    Expects(index < this->validPointsMapSize);
    return {static_cast<int>(index % this->width), static_cast<int>(index / this->width)};
}

template <typename BoundaryConditions>
size_t
ImageMoveFilter<BoundaryConditions>
    ::integerPointToIndex(IntegerPoint point) const
{
    return point.x + this->width * point.y;
}

#if CUDA_DEVICE_COMPILATION

    template <typename BoundaryConditions>
    float
    ImageMoveFilter<BoundaryConditions>
        ::randomUniformNumber()
    {
        // 1 minus curand_normal, because it samples from (0, 1], and we want [0, 1)
        return 1.f - curand_uniform(&(this->states[CUDA_THREAD_IDX]));
    }

#else // CUDA_HOST_COMPILATION

    template <typename BoundaryConditions>
    float
    ImageMoveFilter<BoundaryConditions>
        ::randomUniformNumber()
    {
        return this->uniformDistribution(this->randomGenerator);
    }

#endif

#if CUDA_DEVICE_COMPILATION

    template <typename BoundaryConditions>
    IntegerPoint
    ImageMoveFilter<BoundaryConditions>
        ::randomTracerImagePosition()
    {
        IntegerPoint imagePosition;
        do {
            float floatMapIndex = this->randomUniformNumber() * this->validPointsMapSize;
            size_t mapIndex = static_cast<size_t>(floatMapIndex);
            imagePosition = this->indexToIntegerPoint(mapIndex);
        } while(!this->isPrecomputedTracerValid(imagePosition));
        return imagePosition;
    }

#else // CUDA_HOST_COMPILATION

    template <typename BoundaryConditions>
    IntegerPoint
    ImageMoveFilter<BoundaryConditions>
        ::randomTracerImagePosition()
    {
        float floatCacheIndex = this->randomUniformNumber() * this->validTracerIndicesCache.size();
        size_t cacheIndex = static_cast<size_t>(floatCacheIndex);
        Assert(cacheIndex < this->validTracerIndicesCache.size());
        size_t tracerIndex = this->validTracerIndicesCache[cacheIndex];
        return this->indexToIntegerPoint(tracerIndex);
    }

#endif

template <typename BoundaryConditions>
bool
ImageMoveFilter<BoundaryConditions>
    ::isMoveValid(Tracer tracer, Move move) const
{
    Point from = tracer.getPosition();
    Point to = from + move;
    IntegerPoint integerFrom(from);
    IntegerPoint integerTo(to);

    if (integerFrom == integerTo)
        return true;

    if (!this->isPrecomputedTracerValid(integerTo))
        return false;

    return this->isPrecomputedTracerLineValid(integerFrom, integerTo);
}

template <typename BoundaryConditions>
Tracer
ImageMoveFilter<BoundaryConditions>
    ::randomValidTracer()
{
    IntegerPoint imagePosition = this->randomTracerImagePosition();
    float pixelOffsetX = this->randomUniformNumber();
    float pixelOffsetY = this->randomUniformNumber();

    Point tracerPosition = {imagePosition.x + pixelOffsetX, imagePosition.y + pixelOffsetY};
    return Tracer(tracerPosition, this->tracerRadius);
}

#if CUDA_DEVICE_COMPILATION

    template <typename BoundaryConditions>
    void
    ImageMoveFilter<BoundaryConditions>
        ::setupForTracerRadius(float radius)
    {
        int i = CUDA_THREAD_IDX;
        if (i >= this->validPointsMapSize)
            return;

        this->tracerRadius = radius;
        this->validTracersMap[i] = this->isNotPrecomputedTracerValid(this->indexToIntegerPoint(i), radius);
    }

#else // CUDA_HOST_COMPILATION

    template <typename BoundaryConditions>
    void
    ImageMoveFilter<BoundaryConditions>
        ::setupForTracerRadius(float radius)
    {
        Expects(radius >= 0.f);
        this->tracerRadius = radius;

        this->validTracerIndicesCache.clear();
        for (size_t i = 0; i < this->validPointsMapSize; i++) {
            if (this->isNotPrecomputedTracerValid(this->indexToIntegerPoint(i), radius)) {
                this->validTracersMap[i] = true;
                this->validTracerIndicesCache.push_back(i);
            } else {
                this->validTracersMap[i] = false;
            }
        }

        if (this->validTracerIndicesCache.empty())
            throw std::runtime_error("No valid points found in a given image");
    }

#endif

template <typename BoundaryConditions>
size_t
ImageMoveFilter<BoundaryConditions>
    ::getNumberOfAllPoints() const
{
    return this->validPointsMapSize;
}

#if CUDA_HOST_COMPILATION

    template <typename BoundaryConditions>
    size_t
    ImageMoveFilter<BoundaryConditions>
        ::getNumberOfValidTracers()
    {
        return this->validTracerIndicesCache.size();
    }

#endif /* HOST_COMPILATION */
