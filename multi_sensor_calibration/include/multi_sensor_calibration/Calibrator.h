#ifndef MULTI_SENSOR_CALIBRATION_CALIBRATOR_H
#define MULTI_SENSOR_CALIBRATION_CALIBRATOR_H

#include "CalibrationDataIO.h"
#include "visual_marker_mapping/Camera.h"
#include "visual_marker_mapping/TagDetector.h"
#include "visual_marker_mapping/TagReconstructor.h"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace multi_sensor_calibration
{
struct SensorCalibrationProblem
{

    ceres::Problem optim_problem;

    int total_laser_correspondences;

    std::vector<std::function<double()> > repErrorFns;
    std::map<int, std::vector<std::function<double()> > > repErrorFnsByCam;

    double computeRMSE() const
    {
        if (repErrorFns.empty())
            return 0.0;

        double rms = 0;
        for (const auto& errFn : repErrorFns)
        {
            const double sqrError = errFn();
            // if (sqrt(sqrError)>2)
            // std::cout << "RepError: " << sqrt(sqrError) << std::endl;
            rms += sqrError;
        }
        return sqrt(rms / repErrorFns.size());
    }
    double computeMedianError() const
    {
        if (repErrorFns.empty())
            return 0.0;

        std::vector<double> errors;
        for (const auto& errFn : repErrorFns)
        {
            const double sqrError = errFn();
            errors.push_back(sqrError);
        }
        std::sort(errors.begin(), errors.end());
        return sqrt(errors[errors.size() / 2]);
    }
    double computeRMSEByCam(int cam)
    {
        if (repErrorFnsByCam[cam].empty())
            return 0.0;

        double rms = 0;
        for (const auto& errFn : repErrorFnsByCam[cam])
        {
            const double sqrError = errFn();
            // std::cout << "RepError: " << sqrt(sqrError) << std::endl;
            rms += sqrError;
        }
        return sqrt(rms / repErrorFnsByCam[cam].size());
    }
};
struct Calibrator
{
    Calibrator(CalibrationData calib_data);

    void calibrate(const std::string& visualization_filename);


    enum class OptimizationMode
    {
        SIMPLE_THEN_FULL,
        ONLY_SIMPLE,
        ONLY_FULL
    };

    void optimizeUpToJoint(OptimizationMode optimizationMode);

    CalibrationData calib_data;

    bool computeRelativeCameraPoseFromImg(size_t camera_id, size_t calibration_frame_id,
        const Eigen::Matrix3d& K, const Eigen::Matrix<double, 5, 1>& distCoefficients,
        Eigen::Quaterniond& q, Eigen::Vector3d& t);

    void exportCalibrationResults(const std::string& filePath) const;

    void addSimpleSensorResiduals(
        SensorCalibrationProblem& problem, int calib_frame_id, int sensor_id, bool simple);


    /////////

    // frame, camera -> camera model
    std::map<std::pair<size_t, size_t>, Eigen::Matrix<double, 7, 1> > reconstructedPoses;

    struct JointData
    {
        Eigen::Matrix<double, 7, 1> parent_to_joint_pose;
    };

    std::vector<JointData> jointData;

    void addJointParameterBlocks(SensorCalibrationProblem& problem, int joint_id);


    std::map<int, Eigen::Matrix<double, 7, 1> > location_id_to_location;

private:
    std::vector<std::size_t> pathFromRootToJoint(const std::string& start);
};
}

#endif
