//
//  Copyright (c) 2013 Vladimir Chalupecky
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#include <tps/tps.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <fstream>
#include <ctime>

typedef tps::Thin_plate_spline_transformation< 3, 3 >  Transformation;
typedef Transformation::Domain_point Domain_point;
typedef Transformation::Range_point  Range_point;

int main()
{
    std::vector< Domain_point > dp;
    dp.push_back(Domain_point(0, 0, 0));
    dp.push_back(Domain_point(1, 0, 0));
    dp.push_back(Domain_point(0, 1, 0));
    dp.push_back(Domain_point(1, 1, 0));
    dp.push_back(Domain_point(0, 0, 1));
    dp.push_back(Domain_point(1, 0, 1));
    dp.push_back(Domain_point(0, 1, 1));
    dp.push_back(Domain_point(1, 1, 1));

    boost::variate_generator<boost::mt19937, boost::normal_distribution<> >
            gen(boost::mt19937(time(0)), boost::normal_distribution<>(0.0, 0.2));

    std::vector< Range_point >  rp;
    rp.push_back(Range_point(0 + gen(), 0 + gen(), 0 + gen()));
    rp.push_back(Range_point(1 + gen(), 0 + gen(), 0 + gen()));
    rp.push_back(Range_point(0 + gen(), 1 + gen(), 0 + gen()));
    rp.push_back(Range_point(1 + gen(), 1 + gen(), 0 + gen()));
    rp.push_back(Range_point(0 + gen(), 0 + gen(), 1 + gen()));
    rp.push_back(Range_point(1 + gen(), 0 + gen(), 1 + gen()));
    rp.push_back(Range_point(0 + gen(), 1 + gen(), 1 + gen()));
    rp.push_back(Range_point(1 + gen(), 1 + gen(), 1 + gen()));

    Transformation tps(dp.begin(), dp.end(), rp.begin(), rp.end());
    std::cout << "Integral bending norm: " << tps.integral_bending_norm() << std::endl;

    std::ofstream tpsfile("tps_test.vtk");
    tpsfile << "# vtk DataFile Version 2.0\n";
    tpsfile << "tps\n";
    tpsfile << "ASCII\nDATASET POLYDATA\n";
    tpsfile << "POINTS " << 11 * 11 * 11 << " float\n";

    for (std::size_t i = 0; i < 11; ++i)
    {
        for (std::size_t j = 0; j < 11; ++j)
        {
            for (std::size_t k = 0; k < 11; ++k)
            {
                Domain_point p(i * 0.1, j * 0.1, k * 0.1);
                Range_point r = tps.transform(p);
                tpsfile << r(0) << " " << r(1) << " " << r(2) << std::endl;
            }
        }
    }

    return 0;
}
