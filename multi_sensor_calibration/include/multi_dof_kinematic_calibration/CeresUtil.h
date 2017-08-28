#ifndef MULTI_DOF_KINEMATIC_CALIBRATION_CERESUTIL_H
#define MULTI_DOF_KINEMATIC_CALIBRATION_CERESUTIL_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

template <typename T> inline Eigen::Map<const Eigen::Matrix<T, 2, 1> > eMap2(const T* v)
{
    return Eigen::Map<const Eigen::Matrix<T, 2, 1> >(v);
}
template <typename T> inline Eigen::Map<const Eigen::Matrix<T, 3, 1> > eMap3(const T* v)
{
    return Eigen::Map<const Eigen::Matrix<T, 3, 1> >(v);
}
template <typename T> inline Eigen::Map<const Eigen::Matrix<T, 4, 1> > eMap4(const T* v)
{
    return Eigen::Map<const Eigen::Matrix<T, 4, 1> >(v);
}
template <typename T> inline Eigen::Map<const Eigen::Matrix<T, 6, 1> > eMap6(const T* v)
{
    return Eigen::Map<const Eigen::Matrix<T, 6, 1> >(v);
}
template <typename T> inline Eigen::Map<const Eigen::Quaternion<T> > eMapq(const T* v)
{
    return Eigen::Map<const Eigen::Quaternion<T> >(v);
}

template <typename T> inline Eigen::Map<Eigen::Matrix<T, 2, 1> > eMap2(T* v)
{
    return Eigen::Map<Eigen::Matrix<T, 2, 1> >(v);
}
template <typename T> inline Eigen::Map<Eigen::Matrix<T, 3, 1> > eMap3(T* v)
{
    return Eigen::Map<Eigen::Matrix<T, 3, 1> >(v);
}
template <typename T> inline Eigen::Map<Eigen::Matrix<T, 4, 1> > eMap4(T* v)
{
    return Eigen::Map<Eigen::Matrix<T, 4, 1> >(v);
}
template <typename T> inline Eigen::Map<Eigen::Matrix<T, 6, 1> > eMap6(T* v)
{
    return Eigen::Map<Eigen::Matrix<T, 6, 1> >(v);
}
template <typename T> inline Eigen::Map<Eigen::Quaternion<T> > eMapq(T* v)
{
    return Eigen::Map<Eigen::Quaternion<T> >(v);
}

template <typename T>
void poseAccum(const Eigen::Matrix<T, 4, 1>& q1, const Eigen::Matrix<T, 3, 1>& t1,
    const Eigen::Matrix<T, 4, 1>& q2, const Eigen::Matrix<T, 3, 1>& t2, Eigen::Matrix<T, 4, 1>& q3,
    Eigen::Matrix<T, 3, 1>& t3)
{
    ceres::QuaternionProduct(&q1(0), &q2(0), &q3(0));

    ceres::QuaternionRotatePoint(&q1(0), &t2(0), &t3(0));
    t3 += t1;
}

template <typename T>
Eigen::Matrix<T, 7, 1> cmakePose(const Eigen::Matrix<T, 3, 1>& xyz, const Eigen::Matrix<T, 4, 1>& q)
{
    Eigen::Matrix<T, 7, 1> ret;
    ret << xyz(0), xyz(1), xyz(2), q(0), q(1), q(2), q(3);
    return ret;
}

template <typename T>
Eigen::Matrix<T, 7, 1> cmakePose(const Eigen::Matrix<T, 3, 1>& xyz, const Eigen::Quaternion<T>& q)
{
    Eigen::Matrix<T, 7, 1> ret;
    ret << xyz(0), xyz(1), xyz(2), q.w(), q.x(), q.y(), q.z();
    return ret;
}

template <typename T>
Eigen::Matrix<T, 7, 1> cposeAdd(const Eigen::Matrix<T, 7, 1>& a, const Eigen::Matrix<T, 7, 1>& b)
{
    Eigen::Matrix<T, 7, 1> ret;
    ceres::QuaternionProduct(&a(3), &b(3), &ret(3));
    ceres::QuaternionRotatePoint(&a(3), &b(0), &ret(0));
    ret.template segment<3>(0) += a.template segment<3>(0);
    return ret;
}

template <typename T> Eigen::Matrix<T, 4, 1> cquatConjugate(const Eigen::Matrix<T, 4, 1>& q)
{
    Eigen::Matrix<T, 4, 1> ret;
    ret << q(0), -q(1), -q(2), -q(3);
    return ret;
}

template <typename T> Eigen::Matrix<T, 7, 1> cposeInv(const Eigen::Matrix<T, 7, 1>& a)
{
    Eigen::Matrix<T, 7, 1> ret;
    ret.template segment<4>(3) = cquatConjugate<T>(a.template segment<4>(3));
    ceres::QuaternionRotatePoint(&ret(3), &a(0), &ret(0));
    ret.template segment<3>(0) *= T(-1);
    return ret;
}


template <typename T>
Eigen::Matrix<T, 6, 1> cposeManifoldMinus(
    const Eigen::Matrix<T, 7, 1>& a, const Eigen::Matrix<T, 7, 1>& b)
{
    Eigen::Matrix<T, 6, 1> ret;
    ret.template segment<3>(0) = a.template segment<3>(0) - b.template segment<3>(0);

    Eigen::Matrix<T, 4, 1> bqinv = cquatConjugate<T>(b.template segment<4>(3));

    Eigen::Matrix<T, 4, 1> delta_q;
    ceres::QuaternionProduct(&a(3), &bqinv(0), &delta_q(0));

    ceres::QuaternionToAngleAxis(&delta_q(0), &ret(3));
    return ret;
}


template <typename T>
Eigen::Matrix<T, 3, 1> cposeTransformPoint(
    const Eigen::Matrix<T, 7, 1>& a, const Eigen::Matrix<T, 3, 1>& pt)
{
    Eigen::Matrix<T, 3, 1> ret;
    ceres::UnitQuaternionRotatePoint(&a(3), &pt(0), &ret(0));
    ret += a.template segment<3>(0);
    return ret;
}


//-----------------------------------------------------------------------------

struct GaussianPrior1D
{
    GaussianPrior1D(double mu, double sigma)
        : mu(mu)
        , sigma(sigma)
    {
    }

    template <typename T> bool operator()(const T* const value, T* residuals) const
    {
        // Gauss
        residuals[0] = (*value - T(mu)) / T(sigma);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double mu, const double sigma)
    {
        return new ceres::AutoDiffCostFunction<GaussianPrior1D, 1, 1>(
            new GaussianPrior1D(mu, sigma));
    }

    double mu;
    double sigma;
};


struct OpenCVReprojectionError
{
    OpenCVReprojectionError(const Eigen::Vector2d& observation,
        const Eigen::Matrix<double, 5, 1>& d, const Eigen::Matrix3d& K)
        : observation(observation)
        , d(d)
        , K(K)
    {
    }

    template <typename T>
    bool operator()(const T* const camera_xyz, const T* const camera_q, const T* const point_xyz,
        T* residuals) const
    {

        // transform into camera coordinate system
        T tagCorC[3];
        ceres::UnitQuaternionRotatePoint(camera_q, point_xyz, tagCorC);

        tagCorC[0] += camera_xyz[0];
        tagCorC[1] += camera_xyz[1];
        tagCorC[2] += camera_xyz[2];

        //
        tagCorC[0] = tagCorC[0] / tagCorC[2];
        tagCorC[1] = tagCorC[1] / tagCorC[2];

        // radius^2
        T r2 = tagCorC[0] * tagCorC[0] + tagCorC[1] * tagCorC[1];


        T k1 = T(d(0, 0));
        T k2 = T(d(1, 0));
        T p1 = T(d(2, 0));
        T p2 = T(d(3, 0));
        T k3 = T(d(4, 0));

        // distort
        T xp = tagCorC[0];
        T yp = tagCorC[1];
        T xd = xp * (T(1) + r2 * (k1 + r2 * (k2 + r2 * k3))) + T(2) * p1 * xp * yp
            + p2 * (r2 + T(2) * xp * xp);
        T yd = yp * (T(1) + r2 * (k1 + r2 * (k2 + r2 * k3))) + T(2) * p2 * xp * yp
            + p1 * (r2 + T(2) * yp * yp);

        T fx = T(K(0, 0));
        T fy = T(K(1, 1));
        T cx = T(K(0, 2));
        T cy = T(K(1, 2));

        T predicted_x = T(fx) * xd + T(cx);
        T predicted_y = T(fy) * yd + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observation.x());
        residuals[1] = predicted_y - T(observation.y());

        if (print)
        {
            std::cout << "Predicted: " << predicted_x << " " << predicted_y
                      << " Observed: " << observation.x() << " " << observation.y() << std::endl;
        }

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation,
        const Eigen::Matrix<double, 5, 1>& d, const Eigen::Matrix3d& K)
    {
        return new ceres::AutoDiffCostFunction<OpenCVReprojectionError, 2, 3, 4,
            3>( //  residuals, camPose, tagPose
            new OpenCVReprojectionError(observation, d, K));
    }

    Eigen::Vector2d observation;

    // distortion coefficients
    Eigen::Matrix<double, 5, 1> d;
    Eigen::Matrix3d K;
    bool print = false;
};

#endif
