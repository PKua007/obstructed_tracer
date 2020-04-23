/*
 * ImageMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGEMOVEFILTER_H_
#define IMAGEMOVEFILTER_H_

#include <random>
#include <curand_kernel.h>

#include "simulation/MoveFilter.h"
#include "simulation/IntegerPoint.h"

/**
 * @brief A `__host__` `__device__` MoveFilter accepting moves if they do not collide with stuff on the image.
 *
 * Move is treated as a line (or rather line of disk-like tracers - capsule) drawn on the image and is accepted if
 * this line does not contain pixels which belong to obstacles. The class can accept different ImageBoundaryConditions,
 * eg. free and periodic. The pixels available for finite-sized tracers are precomputed, to the performance is not
 * affected by size of a tracer. Most methods are common for CPU or GPU, however there are some divergencies.
 */
template <typename BoundaryConditions>
class ImageMoveFilter: public MoveFilter {
private:

    /* Random number generator is different on GPU and CPU */
#if CUDA_DEVICE_COMPILATION
    curandState *states;
    size_t numberOfStates;
#else // CUDA_HOST_COMPILATION
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    /* The CPU also has valid tracer indices cache calculated once for every tracer radius. GPU is brute-forcing. */
    std::vector<size_t> validTracerIndicesCache{};
#endif

    float tracerRadius{};
    size_t width{};
    size_t height{};
    BoundaryConditions imageBC{};
    bool* validPointsMap{};   // This is calculated from image
    bool* validTracersMap{};  // This is calculated in ImageMoveFilter::setupForTracerRadius
    size_t validPointsMapSize{};

    CUDA_HOSTDEV void initializeGenerators(unsigned long seed, size_t numberOfTrajectories);
    CUDA_HOSTDEV bool isNotPrecomputedTracerValid(IntegerPoint position, float radius) const;
    CUDA_HOSTDEV bool isPointValid(IntegerPoint point) const;
    CUDA_HOSTDEV bool isPrecomputedTracerValid(IntegerPoint position) const;
    CUDA_HOSTDEV bool isPrecomputedTracerLineValid(IntegerPoint from, IntegerPoint to) const;
    CUDA_HOSTDEV IntegerPoint indexToIntegerPoint(std::size_t index) const;
    CUDA_HOSTDEV size_t integerPointToIndex(IntegerPoint point) const;
    CUDA_HOSTDEV float randomUniformNumber();
    CUDA_HOSTDEV IntegerPoint randomTracerImagePosition();

public:
    /**
     * @brief The constructor of the class performing memory initialization.
     *
     * <p>It takes @a intImage data with 0xFFFFFFFF-like colors and creates vaild points map. Only white pixels are
     * assumed to be valid. The image dimensions are installed on @a imageBC passed. In CPU version the class is also
     * prepared for 0 tracer radius, in GPU version unfortunately state is undefined until the invocation of
     * ImageMoveFilter::setupForTracerRadius.
     *
     * <p>On GPU, the constructor is assumed to be invoked only in the first CUDA thread.
     *
     * <p>What is important, while image coordinates start from top left corner and go right and down, the coordinates
     * in valid point map are translated to cartesian, with image left down corner being at (0, 0).
     *
     * @param intImageData the array of 0xFFFFFFFF-like colors
     * @param width width of the image used
     * @param height height of the image used
     * @param seed seed to be used for sampling random valid tracers
     * @param numberOfTrajectories number of trajectories which will be generated in parallel for GPU version to
     * prepare independent random generators for each
     */
    CUDA_HOSTDEV ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height,
                                 unsigned long seed, size_t numberOfTrajectories);

    CUDA_HOSTDEV ImageMoveFilter(const ImageMoveFilter &other) = delete;
    CUDA_HOSTDEV ImageMoveFilter operator=(ImageMoveFilter other) = delete;

    /**
     * @brief The destructor which is assumed to be only invoked in the first CUDA thread.
     */
    CUDA_HOSTDEV ~ImageMoveFilter();

    /**
     * @brief Checks if a line (or rather line of disk-like tracers - capsule) drawn on the image from @a tracer to
     * `tracer + move` does not contain any pixels of obstacles.
     *
     * All pixels should be in the bounds forced by ImageBoundaryConditions passed in the constructor. The boundary
     * condition are also respected when checking the pixels of a capsule (which, after precomputation described in the
     * constructor reduces to a line).
     *
     * @param tracer initial tracer position
     * @param move move to be performed
     * @return true, if move is valid
     */
    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;

    /**
     * @brief Samples random tracer, which does not collide with an obstacle.
     *
     * In case of CPU, it uses valid tracers cache computed in MoveFilter::setupForTracerRadius, so it olny need to
     * sample one random index. In case of GPU, it samples position on the image as long as it hits the first valid
     * position.
     *
     * @return random tracer, which does not collide with an obstacle. The tracer radius is the one passed to
     * MoveFilter::setupForTracerRadius
     */
    CUDA_HOSTDEV Tracer randomValidTracer() override;

    /**
     * @brief Precomputes valid tracer map from valid point map for a given @a radius.
     *
     * The valid point map is generated in the constructor from an image, while valid tracer map is generated in this
     * method and it represents the points in which we can place disk-like tracer without overlapping with an obstacle.
     * In GPU version it is supposed to be run in parallel in as many thread as there are pixels in the image.
     *
     * @param radius the radius of the tracer to be used
     */
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override;

    /**
     * @brief Returns the number of point in valid points map.
     * @return the number of point in valid points map
     */
    CUDA_HOSTDEV size_t getNumberOfAllPoints() const;

#if CUDA_HOST_COMPILATION
    /**
     * @brief `__host__` only method returning the number of valid tracers found after invoking
     * ImageMoveFilter::setupForTracerRadius.
     * @return the number of valid tracers found after invoking ImageMoveFilter::setupForTracerRadius
     */
    std::size_t getNumberOfValidTracers();
#endif
};


#include "ImageMoveFilter.tpp"

#endif /* IMAGEMOVEFILTER_H_ */
