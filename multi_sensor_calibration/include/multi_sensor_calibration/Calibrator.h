#ifndef MULTI_SENSOR_CALIBRATION_CALIBRATOR_H
#define MULTI_SENSOR_CALIBRATION_CALIBRATOR_H

#include "visual_marker_mapping/Camera.h"
#include "visual_marker_mapping/TagReconstructor.h"
#include "visual_marker_mapping/TagDetector.h"
#include "CalibrationDataIO.h"
#include <Eigen/Core>
#include <set>
#include <vector>
#include <string>
#include <map>

namespace multi_sensor_calibration
{
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


    /////////

    // frame, camera -> camera model
    std::map<std::pair<size_t, size_t>, Eigen::Matrix<double, 7, 1> > reconstructedPoses;

    struct JointData
    {
        Eigen::Matrix<double, 7, 1>  parent_to_joint_pose;
    };

    std::vector<JointData> jointData;


    std::map<int, Eigen::Matrix<double, 7, 1> > location_id_to_location;
};
}

#endif
