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

#ifndef TPS_TPS_HPP
#define TPS_TPS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>

#include <boost/assert.hpp>
#include <boost/type_traits/conditional.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace tps { 
namespace detail {

template< int k >
struct Polyharmonic_spline_kernel_odd
{
    double operator()(double r)
    {
        return std::pow(r, k);
    }
};

template< int k >
struct Polyharmonic_spline_kernel_even
{
    double operator()(double r)
    {
        if (r < 1.0)
        {
            return std::pow(r, k - 1) * std::log(std::pow(r, r));
        }

        return std::pow(r, k) * std::log(r);
    }
};

} // namespace detail

template< int k >
struct Polyharmonic_spline_kernel : public boost::conditional < k % 2,
        detail::Polyharmonic_spline_kernel_odd<k>,
        detail::Polyharmonic_spline_kernel_even<k> >::type
{};


template< int Domain_dim, int Range_dim, typename Kernel_function >
class Polyharmonic_spline_transformation
{
public:
    typedef Eigen::Matrix< double, Domain_dim, 1 > Domain_point;
    typedef Eigen::Matrix< double, Range_dim, 1 >  Range_point;

private:
    typedef std::vector< Domain_point > Domain_points_container;
    typedef std::vector< Range_point> Range_points_container;
    typedef Eigen::MatrixXd Matrix;

public:
    template< typename Forward_iterator_1, typename Forward_iterator_2 >
    Polyharmonic_spline_transformation(
            Forward_iterator_1 domain_points_begin,
            Forward_iterator_1 domain_points_end,
            Forward_iterator_2 range_points_begin,
            Forward_iterator_2 range_points_end,
            double max_relative_error = 1.0e-8
    )
        : domain_points_(domain_points_begin, domain_points_end)
        , bending_norm_(0.0)
        , max_relative_error_(max_relative_error)
    {
        if (domain_points_.size() < Domain_dim + 1)
        {
            throw
                std::runtime_error(
                    "Polyharmonic_spline_transformation(): "
                    "Insufficient number of corresponding pairs given"
                );
        }

        assemble_L_matrix();
        set_range_points(range_points_begin, range_points_end);
    }

    Range_point transform(Domain_point p)
    {
        std::size_t N = domain_points_.size();
        std::size_t M = N + 1 + Domain_dim;

        Range_point result = Wa_.block<1, Range_dim>(N, 0).transpose();

        for (std::size_t i = 0; i < Domain_dim; ++i)
        {
            result += p(i) * Wa_.block<1, Range_dim>(N + 1 + i, 0).transpose();
        }

        for (std::size_t i = 0; i < N; ++i)
        {
            if (p != domain_points_[i])
            {
                result += kernel_((domain_points_[i] - p).norm())
                    * Wa_.block<1, Range_dim>(i, 0).transpose();
            }
        }

        return result;
    }

    template < typename Forward_iterator >
    void set_range_points(Forward_iterator begin, Forward_iterator end)
    {
        if (domain_points_.size() != std::distance(begin, end))
        {
            throw
                std::runtime_error(
                    "Polyharmonic_spline_transformation::set_range_points(): "
                    "The number of domain and range points must be the "
                    "same."
                );
        }

        assemble_Vt_matrix(begin, end);
        update_Wa_matrix();
    }

    double integral_bending_norm() const
    {
        return bending_norm_;
    }

private:
    template < typename Forward_iterator >
    void assemble_Vt_matrix(Forward_iterator begin, Forward_iterator end)
    {
        std::size_t N = domain_points_.size();
        std::size_t M = N + 1 + Domain_dim;

        Vt_.resize(M, Range_dim);
        std::size_t i = 0;

        for (Forward_iterator it = begin; it != end; ++it, ++i)
        {
            Vt_.block<1, Range_dim>(i, 0) = *it;
        }

        Vt_.bottomLeftCorner < Domain_dim + 1, Range_dim > ().setZero();
    }


    void assemble_L_matrix()
    {
        std::size_t N = domain_points_.size();
        std::size_t M = N + 1 + Domain_dim;

        L_.resize(M, M);

        for (std::size_t i = 0; i < N; ++i)
        {
            L_(i, i) = 0.0;

            for (std::size_t j = i + 1; j < N; ++j)
            {
                if (domain_points_[i] == domain_points_[j])
                {
                    throw std::runtime_error(
                        "Polyharmonic_spline_transformation::assemble_L_matrix(): "
                        "Degenerate input points");
                }

                double d = (domain_points_[i] - domain_points_[j]).norm();

                if (d < minimum_distance_)
                {
                    std::cerr <<
                        "Warning: Polyharmonic_spline_transformation::assemble_L_matrix(): "
                        "Input points " << i << " and " << j <<
                        " are too close (dist(i,j) < " << minimum_distance_
                        << ')' << std::endl;
                }

                L_(i, j) = L_(j, i) = kernel_(d);
            }
        }

        for (std::size_t i = 0; i < N; ++i)
        {
            L_.block<1, Domain_dim>(i, N + 1) = domain_points_[i].transpose();
            L_.block<Domain_dim, 1>(N + 1, i) = domain_points_[i];
        }

        L_.col(N).setOnes();
        L_.row(N).setOnes();
        L_.bottomRightCorner < Domain_dim + 1, Domain_dim + 1 > ().setZero();
    }

    void update_Wa_matrix()
    {
        std::size_t N = domain_points_.size();
        std::size_t M = N + 1 + Domain_dim;

        Wa_.resize(M, Range_dim);
        Wa_ = L_.fullPivLu().solve(Vt_);
        double relative_error = (L_ * Wa_ - Vt_).norm() / Vt_.norm();

        if (std::isnan(relative_error))
        {
            throw
                std::runtime_error(
                    "Polyharmonic_spline_transformation::update_Wa_matrix(): "
                    "Cannot define transformation (probably degenerate input "
                    "points)"
                );
        }

        if (relative_error > max_relative_error_)
        {
            throw
                std::runtime_error(
                    "Polyharmonic_spline_transformation::update_Wa_matrix(): "
                    "Relative error too large"
                );
        }

        Matrix bending_norm = Matrix::Zero(1, 1);

        for (std::size_t i = 0; i < Range_dim; ++i)
        {
            bending_norm += Wa_.block(0, i, N, 1).transpose()
                * L_.block(0, 0, N, N) 
                * Wa_.block(0, i, N, 1);
        }

        bending_norm_ = bending_norm(0, 0);
    }

    Domain_points_container domain_points_;
    Kernel_function         kernel_;
    Matrix                  L_, Vt_, Wa_;
    double                  bending_norm_, max_relative_error_;

    static double const minimum_distance_ = 1.0e-6;
};

template < int Domain_dim, int Range_dim >
class Thin_plate_spline_transformation : public Polyharmonic_spline_transformation<
                                            Domain_dim,
                                            Range_dim,
                                            Polyharmonic_spline_kernel< 2 > >
{
    typedef Polyharmonic_spline_transformation<
        Domain_dim,
        Range_dim,
        Polyharmonic_spline_kernel< 2 > > Base;

public:
    template< typename Forward_iterator_1, typename Forward_iterator_2 >
    Thin_plate_spline_transformation(
            Forward_iterator_1 domain_points_begin,
            Forward_iterator_1 domain_points_end,
            Forward_iterator_2 range_points_begin,
            Forward_iterator_2 range_points_end,
            double max_relative_error = 1.0e-8
    )
        : Base(domain_points_begin, domain_points_end, range_points_begin,
                range_points_end, max_relative_error)
    {}
};

} // namespace tps

#endif // TPS_TPS_HPP
