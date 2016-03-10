#include "multi_dof_kinematic_calibration/Calibrator.h"

#include <Eigen/Geometry>

#include "visual_marker_mapping/ReconstructionIO.h"
#include "visual_marker_mapping/DetectionIO.h"
#include "visual_marker_mapping/EigenCVConversions.h"
#include "visual_marker_mapping/PropertyTreeUtilities.h"
#include "visual_marker_mapping/CameraUtilities.h"

#include "multi_dof_kinematic_calibration/CeresUtil.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

#include <iostream>
#include <fstream>

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

namespace multi_dof_kinematic_calibration
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


//-----------------------------------------------------------------------------
Calibrator::Calibrator(CalibrationData calib_data)
    : calib_data(calib_data)
{
}
//-----------------------------------------------------------------------------
void Calibrator::optimizeUpToJoint(size_t upTojointIndex)
{
    auto poseInverse = [](const Eigen::Matrix<double, 7, 1>& pose)
    {
        return cposeInv<double>(pose);
    };
    auto poseAdd = [](const Eigen::Matrix<double, 7, 1>& x,
        const Eigen::Matrix<double, 7, 1>& d) -> Eigen::Matrix<double, 7, 1>
    {
        return cposeAdd<double>(x, d);
    };


    ceres::Problem problem_simple;
    ceres::Problem problem_full;

    const std::vector<int> yzconstant_params = { 1, 2 };
    const auto yzconstant_parametrization = new ceres::SubsetParameterization(3, yzconstant_params);
    const auto yzconstant_parametrization2
        = new ceres::SubsetParameterization(3, yzconstant_params);

    const auto quaternion_parameterization = new ceres::QuaternionParameterization;
    const auto quaternion_parameterization2 = new ceres::QuaternionParameterization;

    for (size_t j = 0; j < upTojointIndex; j++)
    {
        auto& joint_to_parent_pose = jointData[j].joint_to_parent_pose;

        if (calib_data.joints[j].type == "1_dof_joint")
        {
            // x always has to be positive; y,z have to be 0
            problem_simple.AddParameterBlock(
                &joint_to_parent_pose(0), 3, yzconstant_parametrization);
            problem_simple.AddParameterBlock(
                &joint_to_parent_pose(3), 4, quaternion_parameterization);

            // problem_simple.SetParameterLowerBound(&joint_to_parent_pose(0), 0, 0);

            problem_simple.AddParameterBlock(&jointData[j].ticks_to_rad, 1);
            // problem_simple.SetParameterBlockConstant(&jointData[j].ticks_to_rad);

            problem_full.AddParameterBlock(
                &joint_to_parent_pose(0), 3, yzconstant_parametrization2);
            problem_full.AddParameterBlock(
                &joint_to_parent_pose(3), 4, quaternion_parameterization2);

            problem_full.AddParameterBlock(&jointData[j].ticks_to_rad, 1);
            // problem_full.SetParameterBlockConstant(&jointData[j].ticks_to_rad);

            // problem_full.SetParameterLowerBound(&joint_to_parent_pose(0), 0, 0);
        }
        else if (calib_data.joints[j].type == "pose")
        {
            problem_simple.AddParameterBlock(&joint_to_parent_pose(0), 3);
            problem_simple.AddParameterBlock(
                &joint_to_parent_pose(3), 4, quaternion_parameterization);

            problem_full.AddParameterBlock(&joint_to_parent_pose(0), 3);
            problem_full.AddParameterBlock(
                &joint_to_parent_pose(3), 4, quaternion_parameterization2);


            problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(0));
            problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(3));
            problem_full.SetParameterBlockConstant(&joint_to_parent_pose(0));
            problem_full.SetParameterBlockConstant(&joint_to_parent_pose(3));
        }
    }


    using DistinctJointIndex = std::vector<size_t>;
    using DistinctLeverIndex = size_t;
    std::vector<DistinctJointIndex> indexToDistinctJoint(calib_data.calib_frames.size());
    std::vector<DistinctLeverIndex> indexToDistinctLever(calib_data.calib_frames.size());

    using JointState = int;
    using LeverState = std::vector<int>;
    std::vector<std::map<JointState, size_t> > distinctJointPositions(calib_data.joints.size());
    std::map<LeverState, DistinctLeverIndex> distinctLeverPositions;

    std::map<size_t, size_t> distinctFrequencies;


    std::map<DistinctLeverIndex, Eigen::Matrix<double, 7, 1> > camPoses;

    std::vector<std::map<size_t, double> > joint_positions(calib_data.joints.size());

    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        if (calib_data.calib_frames[i].location_id != -1)
        {
            auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
            problem_simple.AddParameterBlock(&location(0), 3);
            problem_simple.AddParameterBlock(&location(3), 4, quaternion_parameterization);

            problem_full.AddParameterBlock(&location(0), 3);
            problem_full.AddParameterBlock(&location(3), 4, quaternion_parameterization2);
        }

        const std::vector<int>& jointConfig = calib_data.calib_frames[i].joint_config;

        for (size_t j = 0; j < upTojointIndex; j++) // this loop could be swapped with the prev
        {
#if 0 // one angle variable for each distinct(!) ptu pose
			auto it = distinctJointPositions[j].find(jointConfig[j]);
			if (it == distinctJointPositions[j].end())
			{
				const size_t dj = distinctJointPositions[j].size();
				indexToDistinctJoint[i].push_back(dj);
				distinctJointPositions[j].emplace(jointConfig[j], dj);

				//
				joint_positions[j][dj]
					= jointConfig[j] * 0.051429 / 180.0 * M_PI; //*0.00089779559;
				markerBAProblem.AddParameterBlock(&joint_positions[j][dj], 1);
				markerBAProblem.SetParameterBlockConstant(&joint_positions[j][dj]);

				markerBAProblem_RepError.AddParameterBlock(&joint_positions[j][dj], 1);

				// markerBAProblem_RepError.SetParameterBlockConstant(&joint_positions[j][dj]);
				auto* alphaPrior
					= GaussianPrior1D::Create(joint_positions[j][dj], 0.05 / 180.0 * M_PI);
				markerBAProblem_RepError.AddResidualBlock(
					alphaPrior, nullptr, &joint_positions[j][dj]);
			}
			else
				indexToDistinctJoint[i].push_back(it->second);
#else
            // one angle variable per ptu pose
            indexToDistinctJoint[i].push_back(i);
            distinctJointPositions[j].emplace(jointConfig[j], i);

            //
            joint_positions[j][i] = jointConfig[j];

            if (calib_data.joints[j].type == "1_dof_joint")
            {

                problem_simple.AddParameterBlock(&joint_positions[j][i], 1);
                problem_simple.SetParameterBlockConstant(&joint_positions[j][i]);

                problem_full.AddParameterBlock(&joint_positions[j][i], 1);

                // Set angular noise
                const double noise_in_ticks = calib_data.joints[j].angular_noise_std_dev
                    / calib_data.joints[j].ticks_to_rad;
                if (noise_in_ticks < 1e-8)
                {
                    problem_full.SetParameterBlockConstant(&joint_positions[j][i]);
                }
                else
                {
                    auto anglePrior
                        = GaussianPrior1D::Create(joint_positions[j][i], noise_in_ticks);
                    problem_full.AddResidualBlock(anglePrior, nullptr, &joint_positions[j][i]);
                }
            }
#endif
        }
        {
            // one camera pose for each distinct lever config
            std::vector<int> leverConfig(jointConfig.begin() + upTojointIndex, jointConfig.end());
            // leverConfig.push_back(calib_data.calib_frames[i].camera_id);

            const auto it = distinctLeverPositions.find(leverConfig);
            if (it == distinctLeverPositions.end())
            {
                const size_t distinct_lever_index = distinctLeverPositions.size();
                indexToDistinctLever[i] = distinct_lever_index;
                distinctLeverPositions.emplace(leverConfig, distinct_lever_index);

                distinctFrequencies[distinct_lever_index] = 1;

                // std::cout << "New Config: ";
                // for (auto i : leverConfig)
                //     std::cout << i << " , ";
                // std::cout << std::endl;

                camPoses[distinct_lever_index] << 0, 0, 0, 1, 0, 0, 0;
                problem_simple.AddParameterBlock(&camPoses[distinct_lever_index](0), 3);
                problem_simple.AddParameterBlock(
                    &camPoses[distinct_lever_index](3), 4, quaternion_parameterization);

                problem_full.AddParameterBlock(&camPoses[distinct_lever_index](0), 3);
                problem_full.AddParameterBlock(
                    &camPoses[distinct_lever_index](3), 4, quaternion_parameterization2);
            }
            else
            {
                indexToDistinctLever[i] = it->second;
                // std::cout << "Existing Config: ";
                // for (auto i : leverConfig)
                //     std::cout << i << " , ";
                // std::cout << std::endl;
                distinctFrequencies[it->second]++;
            }
        }
    }


    std::vector<std::function<double()> > repErrorFns;


    // curImage=0;
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        // only accept lever groups with at least 2 calibration frames
        if (distinctFrequencies[indexToDistinctLever[i]] < 2)
        {
            // std::cout << "Skipping" << std::endl;
            continue;
        }
        // std::cout << "LeverGroupSize: " << distinctFrequencies[indexToDistinctLever[i]]  <<
        // std::endl;

        for (const auto& id_to_cam_model : calib_data.cameraModelById)
        {
            const int camera_id = id_to_cam_model.first;
            const auto& cam_model = id_to_cam_model.second;
            if (camera_id != 1)
                continue;


            const Eigen::Matrix<double, 7, 1> world_to_cam
                = reconstructedPoses[std::make_pair(i, camera_id)];
            // Eigen::Matrix<double, 7, 1> cam_to_world = poseInverse(world_to_cam);
            // std::cout << "CamToWorld : " << cam_to_world.transpose() << std::endl;

            TransformationChain chain;


            /// hier optional eine location node hinzufuegen

            constexpr bool robustify = false;
            std::vector<double*> parameter_blocks;
            for (size_t j = 0; j < upTojointIndex; j++)
            {
                const size_t dj = indexToDistinctJoint[i][j];

                if (calib_data.joints[j].type == "1_dof_joint")
                {
                    chain.add1DOFJoint();

                    parameter_blocks.push_back(&jointData[j].joint_to_parent_pose(0));
                    parameter_blocks.push_back(&jointData[j].joint_to_parent_pose(3));
                    parameter_blocks.push_back(&joint_positions[j][dj]);
                    parameter_blocks.push_back(&jointData[j].ticks_to_rad);
                }
                else if (calib_data.joints[j].type == "pose")
                {
                    chain.addPose();

                    parameter_blocks.push_back(&jointData[j].joint_to_parent_pose(0));
                    parameter_blocks.push_back(&jointData[j].joint_to_parent_pose(3));
                }
            }
            chain.addPose();
            const size_t dl = indexToDistinctLever[i];
            parameter_blocks.push_back(&camPoses[dl](0));
            parameter_blocks.push_back(&camPoses[dl](3));
            auto simpleCostFn = KinematicChainPoseError::Create(world_to_cam, chain);
            problem_simple.AddResidualBlock(simpleCostFn,
                robustify ? new ceres::HuberLoss(1.0) : nullptr, // new ceres::CauchyLoss(3),
                parameter_blocks);

            const auto& camera_observations
                = calib_data.calib_frames[i].cam_id_to_observations[camera_id];
            const auto& world_points = calib_data.reconstructed_map_points;
            iterateMatches(camera_observations, world_points,
                [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp)
                {
                    // check origin pose
                    // {
                    //     OpenCVReprojectionError repErr(tagObs.corners[c],
                    //     camModel.distortionCoefficients,camModel.getK());
                    //     repErr.print=true;
                    //     double res[2];
                    //     repErr(&world_to_cam(0), &world_to_cam(3), &tagCorners[c](0), res);
                    //     std::cout << "ERR: " << sqrt(res[0]*res[0]+res[1]*res[1]) << std::endl;
                    // }

                    auto fullCostFn = KinematicChainRepError::Create(
                        cp, wp, cam_model.distortionCoefficients, cam_model.getK(), chain);
                    problem_full.AddResidualBlock(fullCostFn,
                        robustify ? new ceres::HuberLoss(1.0)
                                  : nullptr, // new ceres::CauchyLoss(3),
                        parameter_blocks);

                    repErrorFns.push_back([parameter_blocks, fullCostFn]() -> double
                        {
                            Eigen::Vector2d err;
                            fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                            return err.squaredNorm();
                        });
                });
        }
    }
    std::cout << "Done creating problems..." << std::endl;

    auto computeRMSE = [&repErrorFns]()
    {
        double rms = 0;
        for (const auto& errFn : repErrorFns)
        {
            const double sqrError = errFn();
            // std::cout << "RepError: " << sqrt(sqrError) << std::endl;
            rms += sqrError;
        }
        return sqrt(rms / repErrorFns.size());
    };

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.num_linear_solver_threads = 4;

#if 0
    for (int j=0;j<jointIndex;j++)
    {
        auto& joint_to_parent_pose = jointData[j].joint_to_parent_pose;
        problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(0));
        problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(3));
    }

    ceres::Solver::Summary summary_pre;
    Solve(options, &problem_simple, &summary_pre);
    std::cout << "Rough Pre Solution: " << summary_pre.termination_type << std::endl;

    for (int j=0;j<jointIndex;j++)
    {
        auto& joint_to_parent_pose = jointData[j].joint_to_parent_pose;
        problem_simple.SetParameterBlockVariable(&joint_to_parent_pose(0));
        problem_simple.SetParameterBlockVariable(&joint_to_parent_pose(3));
    }
#endif


    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_simple, &summary);
    std::cout << "Simple Solution: " << summary.termination_type << std::endl;
    // std::cout << summary.FullReport() << std::endl;

    std::cout << "Training Reprojection Error RMS: " << computeRMSE() << std::endl;


    ceres::Solver::Summary summary2;
    ceres::Solve(options, &problem_full, &summary2);
    std::cout << "Full Solution: " << summary2.termination_type << std::endl;
    std::cout << summary2.FullReport() << std::endl;

    std::cout << "Training Reprojection Error RMS: " << computeRMSE() << std::endl;


    // Print some results
    std::cout << "Resulting Parameters:" << std::endl;

    std::cout << "Tick2Rad for all joints: ";
    for (size_t j = 0; j < upTojointIndex; j++)
        std::cout << jointData[j].ticks_to_rad << " ";
    std::cout << std::endl;

    std::cout << "Joint poses: \n";
    for (size_t j = 0; j < upTojointIndex; j++)
        std::cout << jointData[j].joint_to_parent_pose.transpose() << "\n";
    std::cout << std::endl;

    std::cout << "CamPoses: " << std::endl;
    for (size_t i = 0; i < camPoses.size(); i++)
        std::cout << camPoses[i].transpose() << std::endl;

    if (upTojointIndex < calib_data.joints.size())
        return;

    cameraPose = camPoses[0];
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        const auto& jointConfig = calib_data.calib_frames[i].joint_config;

        Eigen::Matrix<double, 7, 1> root;
        root << 0, 0, 0, 1, 0, 0, 0;

        for (size_t j = 0; j < upTojointIndex; j++)
        {
            root = poseAdd(root, poseInverse(jointData[j].joint_to_parent_pose));

            const double jointAngle = jointConfig[j] * jointData[j].ticks_to_rad;
            // double t = joint_positions[j][indexToDistinctJoint[i][j]];
            Eigen::Matrix<double, 7, 1> tiltRot;
            tiltRot << 0, 0, 0, cos(jointAngle / 2.0), 0, 0, sin(jointAngle / 2.0);
            root = poseAdd(root, tiltRot);

            // auto invRet = poseInverse(root);
            //            DebugVis dbg;
            //            dbg.cam.q = invRet.segment<4>(3);
            //            dbg.cam.t = invRet.segment<3>(0);
            //            dbg.type = 1;
            // debugVis.push_back(dbg);
        }

// std::cout << "NumCamPoses" << camPoses.size() << std::endl;
#if 0
        for (size_t c = 0; c < camPoses.size(); c++)
        {
            auto ret = poseAdd(root, poseInverse(camPoses[c]));
            // std::cout << ret.transpose() << std::endl;
            auto invRet = poseInverse(ret);
            DebugVis dbg;
            dbg.cam.q = invRet.segment<4>(3);
            dbg.cam.t = invRet.segment<3>(0);

            debugVis.push_back(dbg);
        }
#endif
    }

#if 0
    Eigen::Matrix<double, 7, 1> root;
    root << 0, 0, 0, 1, 0, 0, 0;

    for (size_t j = 0; j < jointIndex + 1; j++)
    {
        root = poseAdd(root, poseInverse(jointData[j].joint_to_parent_pose));

        auto invRet = poseInverse(root);
        DebugVis dbg;
        dbg.cam.q = invRet.segment<4>(3);
        dbg.cam.t = invRet.segment<3>(0);
        dbg.type = 1;

        debugVis.push_back(dbg);
    }
#endif

    if (1)
    {
        // compute rms for forward kinematics
        for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
        {
            const auto& jointConfig = calib_data.calib_frames[i].joint_config;

            for (size_t j = 0; j < upTojointIndex; j++)
                joint_positions[j][i] = jointConfig[j];
        }
        std::cout << "Test Reprojection Error RMS: " << computeRMSE() << std::endl;
    }
}
//-----------------------------------------------------------------------------
void Calibrator::exportCalibrationResults(const std::string& filePath) const
{
    namespace pt = boost::property_tree;
    pt::ptree root;

    pt::ptree kinematicChainPt;
    size_t j = 0;
    for (const auto& joint : jointData)
    {
        const Eigen::Vector3d translation = joint.joint_to_parent_pose.head(3);
        const Eigen::Vector4d rotation = joint.joint_to_parent_pose.segment<4>(3);
        const pt::ptree translationPt
            = visual_marker_mapping::matrix2PropertyTreeEigen(translation);
        const pt::ptree rotationPt = visual_marker_mapping::matrix2PropertyTreeEigen(rotation);

        pt::ptree jointDataPt;
        jointDataPt.add_child("translation", translationPt);
        jointDataPt.add_child("rotation", rotationPt);
        jointDataPt.put("ticks_to_rad", joint.ticks_to_rad);
        jointDataPt.put("name", calib_data.joints[j].name);

        kinematicChainPt.push_back(std::make_pair("", jointDataPt));
        j++;
    }
    root.add_child("kinematic_chain", kinematicChainPt);
    pt::ptree cameraPt;
    for (const auto& id_to_cam_model : calib_data.cameraModelById)
    {
        const Eigen::Vector3d translation = cameraPose.head(3);
        const Eigen::Vector4d rotation = cameraPose.segment<4>(3);

        const pt::ptree translationPt
            = visual_marker_mapping::matrix2PropertyTreeEigen(translation);
        const pt::ptree rotationPt = visual_marker_mapping::matrix2PropertyTreeEigen(rotation);

        pt::ptree camDataPt;
        camDataPt.add_child("translation", translationPt);
        camDataPt.add_child("rotation", rotationPt);
        camDataPt.put("id", id_to_cam_model.first);

        cameraPt.push_back(std::make_pair("", camDataPt));
    }
    root.add_child("camera_poses", cameraPt);


    boost::property_tree::write_json(filePath, root);
}
//-----------------------------------------------------------------------------
void Calibrator::calibrate()
{
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        if (calib_data.calib_frames[i].location_id != -1)
        {
            location_id_to_location[calib_data.calib_frames[i].location_id] << 0, 0, 0, 1, 0, 0, 0;
        }

        for (const auto& id_to_cam_model : calib_data.cameraModelById)
        {
            const size_t camera_id = id_to_cam_model.first;
            const auto& cam_model = id_to_cam_model.second;

            Eigen::Quaterniond q;
            Eigen::Vector3d t;
            computeRelativeCameraPoseFromImg(
                camera_id, i, cam_model.getK(), cam_model.distortionCoefficients, q, t);


            const auto cam_pose = cmakePose<double>(t, q);

            //            DebugVis dbg;
            //            dbg.cam.setQuat(q);
            //            dbg.cam.t = t;
            //      if (ptuData.ptuImagePoses[i].cameraId==0)
            //        debugVis.push_back(dbg);

            reconstructedPoses[std::make_pair(i, camera_id)] = cam_pose;
        }
    }

    jointData.resize(calib_data.joints.size());
    for (size_t j = 0; j < jointData.size(); j++)
    {
        jointData[j].joint_to_parent_pose = calib_data.joints[j].joint_to_parent_guess;
        jointData[j].ticks_to_rad = calib_data.joints[j].ticks_to_rad;
    }
    for (size_t j = 0; j < calib_data.joints.size(); j++)
    {
        std::cout << "Optimizing joint: " << calib_data.joints[j].name << std::endl;
        optimizeUpToJoint(j + 1);
        // continue;
        // return;
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
        [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp)
        {
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
