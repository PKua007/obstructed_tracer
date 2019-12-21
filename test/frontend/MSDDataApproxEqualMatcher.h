/*
 * MSDDataApproxEqualMatcher.h
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */

#ifndef MSDDATAAPPROXEQUALMATCHER_H_
#define MSDDATAAPPROXEQUALMATCHER_H_

#include <catch2/catch.hpp>
#include <algorithm>

class MSDDataApproxEqualMatcher : public Catch::MatcherBase<MSDData> {
private:
    const MSDData &expected;
    double epsilon;

public:
    MSDDataApproxEqualMatcher(const MSDData &expected, double epsilon)
            : expected(std::move(expected)), epsilon(epsilon)
    { }

    bool match(const MSDData &actual) const override {
        if (actual.size() != expected.size())
            return false;

        return std::equal(actual.begin(), actual.end(), expected.begin(),
                          [](MSDData::Entry e1, MSDData::Entry e2) {
                              return e1.x == Approx(e2.x)
                                     && e1.y == Approx(e2.y)
                                     && e1.x2 == Approx(e2.x2)
                                     && e1.y2 == Approx(e2.y2)
                                     && e1.xy == Approx(e2.xy);
                          });
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "is, within " << epsilon << " tolerance threshold, equal to" << std::endl << this->expected;
        return ss.str();
    }
};

inline MSDDataApproxEqualMatcher IsApproxEqual(const MSDData &expected, double epsilon) {
    return MSDDataApproxEqualMatcher(expected, epsilon);
}


#endif /* MSDDATAAPPROXEQUALMATCHER_H_ */
