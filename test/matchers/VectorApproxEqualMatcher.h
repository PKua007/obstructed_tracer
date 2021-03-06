/*
 * VectorApproxEqualMatcher.h
 *
 *  Created on: 29 kwi 2020
 *      Author: pkua
 */

#ifndef VECTORAPPROXEQUALMATCHER_H_
#define VECTORAPPROXEQUALMATCHER_H_


#include <catch2/catch.hpp>
#include <sstream>
#include <vector>
#include <iterator>

template<typename T>
class VectorApproxEqualMatcher : public Catch::MatcherBase<std::vector<T>> {
private:
    std::vector<T> expected;
    double epsilon;

public:
    VectorApproxEqualMatcher(std::vector<T> expected, double epsilon)
            : expected(std::move(expected)), epsilon(epsilon)
    { }

    bool match(const std::vector<T> &actual) const override {
        if (this->expected.size() != actual.size())
            return false;
        return std::equal(actual.begin(), actual.end(), this->expected.begin(),
                          [this](T d1, T d2) { return d1 == Approx(d2).epsilon(epsilon); });
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "is, within " << epsilon << " tolerance threshold, equal to" << std::endl;
        std::copy(this->expected.begin(), this->expected.end(), std::ostream_iterator<T>(ss, " "));
        return ss.str();
    }
};

template<typename T>
inline VectorApproxEqualMatcher<T> IsApproxEqual(const std::vector<T> &expected, double epsilon) {
    return VectorApproxEqualMatcher<T>(expected, epsilon);
}


#endif /* VECTORAPPROXEQUALMATCHER_H_ */
