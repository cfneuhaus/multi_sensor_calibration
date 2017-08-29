#include "multi_sensor_calibration/Calibrator.h"

#include <Eigen/Geometry>

#include "visual_marker_mapping/CameraUtilities.h"
#include "visual_marker_mapping/DetectionIO.h"
#include "visual_marker_mapping/EigenCVConversions.h"
#include "visual_marker_mapping/PropertyTreeUtilities.h"
#include "visual_marker_mapping/ReconstructionIO.h"

#include "multi_sensor_calibration/CeresUtil.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

#include "multi_sensor_calibration/DebugVis.h"
#include <fstream>
#include <iostream>

template <typename Map1, typename Map2, typename F>
void iterateMatches(const Map1& m1, const Map2& m2, F&& f)
{
    if (m1.size() < m2.size())
    {
        for (auto it1 = std::begin(m1); it1 != std::end(m1); ++it1)
        {
            const auto it2 = m2.find(it1->first);
            if (it2 == m2.end())
                continue;
            f(it1->first, it1->second, it2->second);
        }
    }
    else
    {
        for (auto it2 = std::begin(m2); it2 != std::end(m2); ++it2)
        {
            const auto it1 = m1.find(it2->first);
            if (it1 == m1.end())
                continue;
            f(it1->first, it1->second, it2->second);
        }
    }
}

namespace multi_sensor_calibration
{
struct TransformationChain
{
    enum class TransformType
    {
        POSE,
        JOINT1DOF
    };
    void addPose() { chain.push_back(TransformType::POSE); }
    void add1DOFJoint() { chain.push_back(TransformType::JOINT1DOF); }

    template <typename CostFn> void addParametersToCostFn(CostFn* cost_function) const
    {
        for (const TransformType& t : chain)
        {
            if (t == TransformType::JOINT1DOF)
            {
                cost_function->AddParameterBlock(3); // trans
                cost_function->AddParameterBlock(4); // quat
                cost_function->AddParameterBlock(1); // angle
                cost_function->AddParameterBlock(1); // joint scale
            }
            else if (t == TransformType::POSE)
            {
                cost_function->AddParameterBlock(3); // trans
                cost_function->AddParameterBlock(4); // quat
            }
        }
    }

    template <typename T> Eigen::Matrix<T, 7, 1> endEffectorPose(T const* const* parameters) const
    {
        Eigen::Matrix<T, 7, 1> world_to_end_pose;
        world_to_end_pose << T(0), T(0), T(0), T(1), T(0), T(0), T(0);
        if (chain.empty())
        {
            return world_to_end_pose;
        }
        int pindex = 0;
        for (size_t i = 0; i < chain.size(); ++i)
        {
            if (chain[i] == TransformType::JOINT1DOF)
            {
                const auto rotaxis_to_parent
                    = cmakePose<T>(eMap3(parameters[pindex + 0]), eMap4(parameters[pindex + 1]));
                const T jointScale = *parameters[pindex + 3];
                const T angle = *parameters[pindex + 2] * jointScale;
                Eigen::Matrix<T, 7, 1> invRot;
                invRot << T(0), T(0), T(0), cos(-angle / T(2)), T(0), T(0), sin(-angle / T(2));

                // world_to_end_pose = cposeAdd(world_to_cam, cposeAdd(invRot, rotaxis_to_parent));

                world_to_end_pose = cposeAdd(
                    invRot, cposeAdd(rotaxis_to_parent,
                                world_to_end_pose)); // cposeAdd(world_to_cam, cposeAdd(invRot,
                // rotaxis_to_parent));

                pindex += 4;
            }
            else if (chain[i] == TransformType::POSE)
            {
                const auto parent_to_pose
                    = cmakePose<T>(eMap3(parameters[pindex + 0]), eMap4(parameters[pindex + 1]));

                world_to_end_pose = cposeAdd(parent_to_pose, world_to_end_pose);

                pindex += 2;
            }
        }
        return world_to_end_pose;
    }

    std::vector<TransformType> chain;
};

struct KinematicChainRepError
{
    KinematicChainRepError(const Eigen::Vector2d& observation, const Eigen::Vector3d& point_3d,
        const Eigen::Matrix<double, 5, 1>& d, const Eigen::Matrix3d& K,
        const TransformationChain& chain)
        : repError(observation, d, K)
        , point_3d(point_3d)
        , chain(chain)
    {
    }

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_cam = chain.endEffectorPose<T>(parameters);

        const Eigen::Matrix<T, 3, 1> point_3d_T = point_3d.cast<T>();
        return repError(&world_to_cam(0), &world_to_cam(3), &point_3d_T(0), residuals);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation,
        const Eigen::Vector3d& point_3d, const Eigen::Matrix<double, 5, 1>& d,
        const Eigen::Matrix3d& K, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<KinematicChainRepError, 4>(
            new KinematicChainRepError(observation, point_3d, d, K, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(2);
        return cost_function;
    }

    OpenCVReprojectionError repError;
    Eigen::Vector3d point_3d;
    TransformationChain chain;
};

struct KinematicChainPoseError
{
    KinematicChainPoseError(
        const Eigen::Matrix<double, 7, 1>& expected_world_to_cam, const TransformationChain& chain)
        : expected_world_to_cam(expected_world_to_cam)
        , chain(chain)
    {
    }

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_cam = chain.endEffectorPose<T>(parameters);
        eMap6(&residuals[0]) = cposeManifoldMinus<T>(world_to_cam, expected_world_to_cam.cast<T>());
        // eMap3(&residuals[3])*=T(1000);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        const Eigen::Matrix<double, 7, 1>& expected_world_to_cam, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<KinematicChainPoseError, 4>(
            new KinematicChainPoseError(expected_world_to_cam, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(6);
        return cost_function;
    }

    Eigen::Matrix<double, 7, 1> expected_world_to_cam;
    TransformationChain chain;
};

struct MarkerPoint2PlaneError
{
    MarkerPoint2PlaneError(const Eigen::Matrix<double, 7, 1>& marker_to_world,
        const Eigen::Vector3d& laser_point, const TransformationChain& chain)
        : marker_to_world(marker_to_world)
        , laser_point(laser_point)
        , chain(chain)
    {
        marker_plane_n = cposeTransformPoint(marker_to_world, Eigen::Vector3d(0, 0, 1))
            - marker_to_world.segment<3>(0);
        marker_plane_n /= marker_plane_n.norm();
        marker_plane_d = marker_plane_n.transpose() * marker_to_world.segment<3>(0);

        //		std::cout << "pos: " << marker_to_world.transpose() << std::endl;
        //		std::cout << "norm: " << marker_plane_n.transpose() << std::endl;
    }


    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_laser = chain.endEffectorPose<T>(parameters);
        auto laser_to_world = cposeInv(world_to_laser);

        Eigen::Matrix<T, 3, 1> laser_point_in_world
            = cposeTransformPoint<T>(laser_to_world, laser_point.cast<T>());

        T td = laser_point_in_world.transpose() * marker_plane_n.cast<T>();

        *residuals = (T(marker_plane_d) - td) / T(0.01);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Matrix<double, 7, 1>& marker_to_world,
        const Eigen::Vector3d& laser_point, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<MarkerPoint2PlaneError, 4>(
            new MarkerPoint2PlaneError(marker_to_world, laser_point, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(1);
        return cost_function;
    }

    Eigen::Matrix<double, 7, 1> marker_to_world;
    Eigen::Vector3d laser_point;

    Eigen::Vector3d marker_plane_n;
    double marker_plane_d;

    TransformationChain chain;
};


//-----------------------------------------------------------------------------
Calibrator::Calibrator(CalibrationData calib_data)
    : calib_data(std::move(calib_data))
{
}
//-----------------------------------------------------------------------------
void Calibrator::optimizeUpToJoint(OptimizationMode optimizationMode)
{
    auto pathFromRootToJoint = [this](const std::string& start) -> std::vector<size_t> {
        std::string cur_joint_name = start;
        std::vector<size_t> joint_list;
        if (cur_joint_name == "base")
            return joint_list;
        size_t cur_index = calib_data.name_to_joint[cur_joint_name];
        joint_list.push_back(cur_index);
        while (1)
        {
            cur_joint_name = calib_data.joints[cur_index].parent;
            if (cur_joint_name == "base")
                break;
            cur_index = calib_data.name_to_joint[cur_joint_name];
            joint_list.push_back(cur_index);
        }
        std::reverse(joint_list.begin(), joint_list.end());
        return joint_list;
    };


    ceres::Problem problem_simple;
    ceres::Problem problem_full;

    const std::vector<int> yzconstant_params = { 1, 2 };
    //    const auto yzconstant_parametrization = new ceres::SubsetParameterization(3,
    //    yzconstant_params);
    //    const auto yzconstant_parametrization2
    //        = new ceres::SubsetParameterization(3, yzconstant_params);

    const auto quaternion_parameterization = new ceres::QuaternionParameterization;
    const auto quaternion_parameterization2 = new ceres::QuaternionParameterization;

    for (size_t j = 0; j < jointData.size(); j++)
    {
        auto& parent_to_joint_pose = jointData[j].parent_to_joint_pose;

        if (calib_data.joints[j].type == "pose")
        {
            std::cout << "add param: " << calib_data.joints[j].name << std::endl;
            problem_full.AddParameterBlock(&parent_to_joint_pose(0), 3);
            problem_full.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization2);
            problem_simple.AddParameterBlock(&parent_to_joint_pose(0), 3);
            problem_simple.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization);


            if (calib_data.joints[j].fixed)
            {
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(3));
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(3));
            }
        }
    }

    std::set<int> initialized_locations;

    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        if (calib_data.calib_frames[i].location_id == -1)
            continue;
        if (!initialized_locations.count(calib_data.calib_frames[i].location_id))
        {
            auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
            problem_full.AddParameterBlock(&location(0), 3);
            problem_full.AddParameterBlock(&location(3), 4, quaternion_parameterization2);
            problem_simple.AddParameterBlock(&location(0), 3);
            problem_simple.AddParameterBlock(&location(3), 4, quaternion_parameterization);

            if (calib_data.optional_location_infos.count(calib_data.calib_frames[i].location_id))
            {
                if (calib_data.optional_location_infos[calib_data.calib_frames[i].location_id]
                        .fixed)
                {
                    problem_full.SetParameterBlockConstant(&location(0));
                    problem_full.SetParameterBlockConstant(&location(3));
                    problem_simple.SetParameterBlockConstant(&location(0));
                    problem_simple.SetParameterBlockConstant(&location(3));
                }
            }

            initialized_locations.insert(calib_data.calib_frames[i].location_id);
        }
    }

    std::vector<std::function<double()> > repErrorFns;
    std::map<int, std::vector<std::function<double()> > > repErrorFnsByCam;


    std::cout << "Building optimization problems..." << std::endl;

    int total_laser_correspondences = 0;

    // curImage=0;
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        for (const auto& sensor_id_to_type : calib_data.sensor_id_to_type)
        {
            const int sensor_id = sensor_id_to_type.first;
            const std::string& sensor_type = sensor_id_to_type.second;

            TransformationChain chain;
            std::vector<double*> parameter_blocks;

            if (calib_data.calib_frames[i].location_id != -1)
            {
                auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
                chain.addPose();

                parameter_blocks.push_back(&location(0));
                parameter_blocks.push_back(&location(3));
            }

            const auto path_to_sensor
                = pathFromRootToJoint(calib_data.sensor_id_to_parent_joint[sensor_id]);

            constexpr bool robustify = true;

            for (size_t pj = 0; pj < path_to_sensor.size(); pj++)
            {
                const size_t j = path_to_sensor[pj];

                if (calib_data.joints[j].type == "pose")
                {
                    chain.addPose();

                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
                }
            }

            if (sensor_type == "camera")
            {
                // sensor is a cam
                const int camera_id = sensor_id;
                const auto& cam_model = calib_data.cameraModelById[camera_id];

                if (reconstructedPoses.count(std::make_pair(i, camera_id)))
                {
                    const Eigen::Matrix<double, 7, 1> world_to_cam
                        = reconstructedPoses[std::make_pair(i, camera_id)];

                    // location_id_to_location[calib_data.calib_frames[i].location_id] =
                    // world_to_cam;
                    // //////// hack

                    auto simpleCostFn = KinematicChainPoseError::Create(world_to_cam, chain);
                    problem_simple.AddResidualBlock(simpleCostFn,
                        robustify ? new ceres::HuberLoss(1.0)
                                  : nullptr, // new ceres::CauchyLoss(3),
                        parameter_blocks);
                }


                // const auto& cam_model = id_to_cam_model.second;
                const auto& camera_observations
                    = calib_data.calib_frames[i].cam_id_to_observations[camera_id];
                const auto& world_points = calib_data.reconstructed_map_points;
                iterateMatches(camera_observations, world_points,
                    [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp) {
                        // check origin pose
                        // {
                        //     OpenCVReprojectionError repErr(tagObs.corners[c],
                        //     camModel.distortionCoefficients,camModel.getK());
                        //     repErr.print=true;
                        //     double res[2];
                        //     repErr(&world_to_cam(0), &world_to_cam(3), &tagCorners[c](0), res);
                        //     std::cout << "ERR: " << sqrt(res[0]*res[0]+res[1]*res[1]) <<
                        //     std::endl;
                        // }


                        auto fullCostFn = KinematicChainRepError::Create(
                            cp, wp, cam_model.distortionCoefficients, cam_model.getK(), chain);
                        problem_full.AddResidualBlock(fullCostFn,
                            robustify ? new ceres::HuberLoss(1.0)
                                      : nullptr, // new ceres::CauchyLoss(3),
                            parameter_blocks);

                        repErrorFns.push_back([parameter_blocks, fullCostFn]() -> double {
                            Eigen::Vector2d err;
                            fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                            return err.squaredNorm();
                        });
                        repErrorFnsByCam[camera_id].push_back(
                            [parameter_blocks, fullCostFn]() -> double {
                                Eigen::Vector2d err;
                                fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                                return err.squaredNorm();
                            });
                    });
            }
            else if (sensor_type == "laser_3d")
            {
                if (optimizationMode == OptimizationMode::ONLY_SIMPLE)
                    continue;
                const auto& cur_scan
                    = calib_data.calib_frames[i].sensor_id_to_laser_scan_3d[sensor_id];

                const Eigen::Matrix<double, 7, 1> world_to_laser
                    = chain.endEffectorPose(&parameter_blocks[0]);
                const auto laser_to_world = cposeInv<double>(world_to_laser);

                int corresp = 0;
                for (int p = 0; p < cur_scan->points.cols(); p++)
                {
                    const Eigen::Vector3d pt = cur_scan->points.col(p);
                    const Eigen::Vector3d ptw = cposeTransformPoint<double>(laser_to_world, pt);

                    const visual_marker_mapping::ReconstructedTag* min_tag = nullptr;
                    double min_sqr_dist = 9999999999.0;
                    for (const auto& id_to_rec_tag : calib_data.reconstructed_tags)
                    {
                        const visual_marker_mapping::ReconstructedTag& tag = id_to_rec_tag.second;
#if 0
	                    Eigen::Matrix<double, 7, 1> marker_to_world;
	                    marker_to_world.segment<3>(0) = tag.t;
	                    marker_to_world.segment<4>(3) = tag.q;
	                    auto world_to_marker = cposeInv<double>(marker_to_world);
	                    auto mpt = cposeTransformPoint<double>(world_to_marker, ptw);
	
	                    if ((mpt.x() < -0.12) || (mpt.y() < -0.12) || (mpt.x() > 0.12)
	                        || (mpt.y() > 0.12))
	                        continue;
	
	                    if (mpt.z() < -0.2)
	                        continue;
	                    if (mpt.z() > 0.2)
	                        continue;
	
	                    const double dist = mpt.z();
#else
                        const double sqr_dist = (tag.t - ptw).squaredNorm();
                        const double marker_radius
                            = std::max(tag.tagWidth, tag.tagHeight) / 2.0 * sqrt(2.0);
#endif
                        if ((sqr_dist < min_sqr_dist) && (sqr_dist < marker_radius * marker_radius))
                        {
                            min_tag = &tag;
                            min_sqr_dist = sqr_dist;
                        }
                    }
                    if (!min_tag)
                        continue;

                    // dbgout.addPoint(ptw);

                    Eigen::Matrix<double, 7, 1> marker_to_world;
                    marker_to_world.segment<3>(0) = min_tag->t;
                    marker_to_world.segment<4>(3) = min_tag->q;

#if 1
                    auto fullCostFn = MarkerPoint2PlaneError::Create(marker_to_world, pt, chain);
                    problem_full.AddResidualBlock(fullCostFn,
                        robustify ? new ceres::HuberLoss(1.0) : nullptr, parameter_blocks);
#endif


                    //			{
                    //            Eigen::Vector3d err;
                    //            double* parameter_blocks[4] = { &world_to_cam_poses[i](0),
                    //            &world_to_cam_poses[i](3) ,&cam_to_laser_pose(0),
                    //											&cam_to_laser_pose(3)};
                    //            fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                    //            std::cout << "err: " << err.transpose() << std::endl;;
                    //			}

                    corresp++;
                    total_laser_correspondences++;
                }
                // std::cout << "Corresp : "<< corresp << std::endl;
            }
        }
    }
    std::cout << "Building optimization problems...done!" << std::endl;
    std::cout << "Laser correspondence count: " << total_laser_correspondences << std::endl;

    auto computeRMSE = [&repErrorFns]() -> double {
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
    };
    auto computeMedianError = [&repErrorFns]() -> double {
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
    };
    auto computeRMSEByCam = [&repErrorFnsByCam](int cam) -> double {
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
    };

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.num_linear_solver_threads = 4;


    if (optimizationMode == OptimizationMode::ONLY_SIMPLE)
    {
        std::cout << "Solving simple optimization problem..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem_simple, &summary);
    }
    if (optimizationMode == OptimizationMode::ONLY_SIMPLE)
    {
    }
    else
    {
        std::cout << "Solving full optimization problem..." << std::endl;
        ceres::Solver::Summary summary2;
        ceres::Solve(options, &problem_full, &summary2);
        std::cout << "    Full optimization returned termination type " << summary2.termination_type
                  << std::endl;
        // std::cout << summary2.FullReport() << std::endl;
    }

    std::cout << "    Full training reprojection error RMS: " << computeRMSE() << " px"
              << " Median: " << computeMedianError() << " px" << std::endl;
    std::cout << "Solving full optimization problem...done!" << std::endl;

    std::cout << std::endl;

    // Print some results
    std::cout << "Resulting Parameters:" << std::endl;

    std::cout << "    Joint poses:\n";
    for (size_t j = 0; j < calib_data.joints.size(); j++)
    {
        std::cout << "        " << calib_data.joints[j].name << ": "
                  << jointData[j].parent_to_joint_pose.transpose() << std::endl;
    }
}
//-----------------------------------------------------------------------------
void Calibrator::exportCalibrationResults(const std::string& filePath) const
{
    namespace pt = boost::property_tree;
    pt::ptree root;

    pt::ptree kinematicChainPt;
    for (size_t j = 0; j < calib_data.joints.size(); ++j)
    {
        const auto pose = jointData[j].parent_to_joint_pose;
        const pt::ptree posePt = visual_marker_mapping::matrix2PropertyTreeEigen(pose);

        const auto pose_guess = calib_data.joints[j].parent_to_joint_guess;
        const pt::ptree poseGuessPt = visual_marker_mapping::matrix2PropertyTreeEigen(pose_guess);

        pt::ptree jointDataPt;
        jointDataPt.add_child("parent_to_joint_pose", posePt);
        jointDataPt.add_child("parent_to_joint_pose_guess", poseGuessPt);
        jointDataPt.put("type", calib_data.joints[j].type);
        jointDataPt.put("name", calib_data.joints[j].name);
        jointDataPt.put("parent", calib_data.joints[j].parent);
        jointDataPt.put("fixed", calib_data.joints[j].fixed ? "true" : "false");

        kinematicChainPt.push_back(std::make_pair("", jointDataPt));
    }
    root.add_child("hierarchy", kinematicChainPt);
    pt::ptree locationPt;
    for (const auto& cur_location_id_to_location : location_id_to_location)
    {
        const int location_id = cur_location_id_to_location.first;
        const auto pose = cur_location_id_to_location.second;
        const pt::ptree posePt = visual_marker_mapping::matrix2PropertyTreeEigen(pose);

        pt::ptree curLocationPt;
        curLocationPt.add_child("world_to_location_pose", posePt);
        curLocationPt.put("location_id", location_id);
        if (calib_data.optional_location_infos.count(location_id))
        {
            const auto& loc_info = calib_data.optional_location_infos.find(location_id)->second;
            const auto initial_pose = loc_info.world_to_location_pose_guess;
            const pt::ptree poseGuessPt
                = visual_marker_mapping::matrix2PropertyTreeEigen(initial_pose);
            curLocationPt.add_child("world_to_location_pos_guess", poseGuessPt);

            curLocationPt.put("fixed", loc_info.fixed ? "true" : "false");
        }

        locationPt.push_back(std::make_pair("", curLocationPt));
    }
    root.add_child("locations", locationPt);
    boost::property_tree::write_json(filePath, root);
}
//-----------------------------------------------------------------------------
void Calibrator::calibrate(const std::string& visualization_filename)
{
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        const int location_id = calib_data.calib_frames[i].location_id;
        if (location_id != -1)
        {
            if (!calib_data.optional_location_infos.count(location_id))
            {
                location_id_to_location[location_id] << 0, 0, 0, 1, 0, 0, 0;
            }
            else
            {
                const auto loc_pose
                    = calib_data.optional_location_infos[location_id].world_to_location_pose_guess;
                location_id_to_location[location_id] = loc_pose;
            }
        }

        for (const auto& id_to_cam_model : calib_data.cameraModelById)
        {
            const size_t camera_id = id_to_cam_model.first;
            const auto& cam_model = id_to_cam_model.second;

            Eigen::Quaterniond q;
            Eigen::Vector3d t;
            bool success = computeRelativeCameraPoseFromImg(
                camera_id, i, cam_model.getK(), cam_model.distortionCoefficients, q, t);
            if (!success)
            {
                std::cerr << "Initialization failed" << std::endl;
            }
            else
            {
                const auto cam_pose = cmakePose<double>(t, q);

                //            DebugVis dbg;
                //            dbg.cam.setQuat(q);
                //            dbg.cam.t = t;
                //      if (ptuData.ptuImagePoses[i].cameraId==0)
                //        debugVis.push_back(dbg);

                reconstructedPoses[std::make_pair(i, camera_id)] = cam_pose;
            }
        }
    }

    jointData.resize(calib_data.joints.size());
    for (size_t j = 0; j < jointData.size(); j++)
    {
        jointData[j].parent_to_joint_pose = calib_data.joints[j].parent_to_joint_guess;
        // HACK: Joint positions hier her?
    }

    size_t start_joint = 0;
    std::vector<std::vector<size_t> > joint_to_children(calib_data.joints.size());
    for (size_t j = 0; j < calib_data.joints.size(); j++)
    {
        if (calib_data.joints[j].parent == "base")
        {
            start_joint = j;
            continue;
        }
        size_t parent_j = calib_data.name_to_joint[calib_data.joints[j].parent];
        joint_to_children[parent_j].push_back(j);
    }

    optimizeUpToJoint(OptimizationMode::ONLY_SIMPLE);
    optimizeUpToJoint(OptimizationMode::ONLY_FULL);
    // when there are laser sensors, iterate the final optimization a couple of times so that the
    // correspondences can improve
    if (!calib_data.laser_sensor_ids.empty())
    {
        for (int its = 0; its < 2; its++)
        {
            optimizeUpToJoint(OptimizationMode::ONLY_FULL);
            std::cout << "-------------------------------------------------------------------------"
                      << std::endl;
            std::cout << "It: " << its << std::endl;
        }
    }

    //////////////////////////////////
    // visualize results
    if (visualization_filename == "")
        return;

    DebugOutput dbg_out(visualization_filename);
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        TransformationChain chain;

        std::vector<double*> parameter_blocks;

        Eigen::Vector3d last_pos(0, 0, 0);

        // Eigen::Matrix<double,7,1> loc;
        // loc << 0,0,0,1,0,0,0;
        if (calib_data.calib_frames[i].location_id != -1)
        {
            auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
            chain.addPose();

            parameter_blocks.push_back(&location(0));
            parameter_blocks.push_back(&location(3));

            dbg_out.addPose(
                location, "location_" + std::to_string(calib_data.calib_frames[i].location_id));

            // std::cout << "Location: " << location.transpose() << std::endl;

            last_pos = cposeInv(location).segment<3>(0);
        }

        std::map<std::string, Eigen::Matrix<double, 7, 1> > world_to_joint_poses;


        std::function<void(
            const Eigen::Vector3d&, TransformationChain, std::vector<double*>, size_t)>
            process = [&](const Eigen::Vector3d& last_posi, TransformationChain cur_chain,
                std::vector<double*> parameter_blocks, size_t j) {
                if (calib_data.joints[j].type == "pose")
                {
                    cur_chain.addPose();

                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
                }

                const auto world_to_pose = cur_chain.endEffectorPose(&parameter_blocks[0]);
                world_to_joint_poses[calib_data.joints[j].name] = world_to_pose;
                // std::cout << "Joint pose of " << calib_data.joints[j].name << "is " <<
                // world_to_pose.transpose() << std::endl;

                dbg_out.addPose(world_to_pose, calib_data.joints[j].name);

                const Eigen::Vector3d cur_pos = cposeInv(world_to_pose).segment<3>(0);

                dbg_out.addLine(last_posi, cur_pos);

                for (size_t cj : joint_to_children[j])
                    process(cur_pos, cur_chain, parameter_blocks, cj);
            };
        process(last_pos, chain, parameter_blocks, start_joint);

#if 1 // Show sensor data
        if (i == 0)
        {
            for (const auto& sensor_id_to_type : calib_data.sensor_id_to_type)
            {
                const int sensor_id = sensor_id_to_type.first;
                if (sensor_id_to_type.second == "laser_3d")
                {
                    const std::string& parent_joint
                        = calib_data.sensor_id_to_parent_joint[sensor_id];
                    auto world_to_sensor = world_to_joint_poses[parent_joint];
                    auto sensor_to_world = cposeInv<double>(world_to_sensor);
                    // std::cout << "sensor_to_world: " << sensor_to_world.transpose() << std::endl;

                    const auto& scan
                        = calib_data.calib_frames[i].sensor_id_to_laser_scan_3d[sensor_id];
                    for (int sp = 0; sp < scan->points.cols(); sp++)
                    {
                        const Eigen::Vector3d pw
                            = cposeTransformPoint<double>(sensor_to_world, scan->points.col(sp));
                        if (!std::isnan(pw.x()))
                            dbg_out.addPoint(pw);
                    }
                }
            }
        }
#endif
    }
}
//-----------------------------------------------------------------------------
bool Calibrator::computeRelativeCameraPoseFromImg(size_t camera_id, size_t calibration_frame_id,
    const Eigen::Matrix3d& K, const Eigen::Matrix<double, 5, 1>& distCoefficients,
    Eigen::Quaterniond& q, Eigen::Vector3d& t)
{
    std::vector<Eigen::Vector3d> markerCorners3D;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > observations2D;
    // find all matches between this image and the reconstructions

    const auto& camera_observations
        = calib_data.calib_frames[calibration_frame_id].cam_id_to_observations[camera_id];
    const auto& world_points = calib_data.reconstructed_map_points;
    iterateMatches(camera_observations, world_points,
        [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp) {
            observations2D.push_back(cp);
            markerCorners3D.push_back(wp);
        });

    std::cout << "   Reconstructing camera pose from " << observations2D.size()
              << " 2d/3d correspondences" << std::endl;

    if (observations2D.empty())
        return false;

    Eigen::Matrix3d R;
    // solvePnPEigen(markerCorners3D, observations2D, K, distCoefficients, R, t);
    visual_marker_mapping::solvePnPRansacEigen(
        markerCorners3D, observations2D, K, distCoefficients, R, t);

    q = Eigen::Quaterniond(R);
    return true;
}
//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
