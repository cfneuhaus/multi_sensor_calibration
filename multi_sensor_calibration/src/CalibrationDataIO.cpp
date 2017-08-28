#include "multi_dof_kinematic_calibration/CalibrationDataIO.h"
#include "visual_marker_mapping/DetectionIO.h"

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "visual_marker_mapping/CameraUtilities.h"
#include "visual_marker_mapping/ReconstructionIO.h"
#include <random>

namespace
{
multi_dof_kinematic_calibration::Scan3D loadScan(const std::string& filename)
{
    multi_dof_kinematic_calibration::Scan3D ret;
    FILE* r = fopen((filename).c_str(), "rb");
    if (!r)
        throw std::runtime_error("Could not open " + filename);
    std::uint32_t numPts;
    fread(&numPts, sizeof(std::uint32_t), 1, r);
    ret.points.resize(3, numPts);
#if 1
    std::vector<double> tmp(4 * numPts);
    fread(&tmp[0], 4 * sizeof(double), numPts, r);
    fclose(r);

    for (size_t i = 0; i < numPts; i++)
        ret.points.col(i) << tmp[4 * i + 0], tmp[4 * i + 1], tmp[4 * i + 2];
#else
    // std::ofstream tmp("out.txt");
    for (int i = 0; i < numPts; i++)
    {
        char buffer[32];
        fread(buffer, 32, 1, r);
        const float* floats = (const float*)&buffer[0];
        //		for (int f=0;f<8;f++)
        //			std::cout << floats[f] << " ";
        //		std::cout << std::endl;
        // ret.points.col(i) << floats[1], -floats[0], floats[2];
        ret.points.col(i) << floats[0], floats[1], floats[2];
        // std::cout << ret.points.col(i).transpose() << std::endl;
        // tmp << ret.points.col(i).transpose() << std::endl;
    }
    fclose(r);
#endif
    //	{
    //	FILE*w=fopen((filename).c_str(),"w");
    //	fwrite(&numPts,4,1,w);
    //	for (int i=0;i<numPts;i++)
    //	{
    //		Eigen::Vector4d tmp;
    //		tmp.segment<3>(0)=ret.points.col(i);
    //		tmp(3)=0;

    //		fwrite(&tmp(0),sizeof(double)*4,1,w);
    //	}
    //	fclose(w);
    //	}
    return ret;
}
void addNoiseToScan(multi_dof_kinematic_calibration::Scan3D& scan, double std_dev)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, std_dev);
    for (int i = 0; i < scan.points.cols(); i++)
        scan.points.col(i) += Eigen::Vector3d(d(gen), d(gen), d(gen));
}
}


namespace multi_dof_kinematic_calibration
{
//----------------------------------------------------------------------------
CalibrationData::CalibrationData(const std::string& filePath, double laser_noise_std_dev)
{
    namespace pt = boost::property_tree;
    pt::ptree rootNode;
    pt::read_json(filePath, rootNode);

    // Read Reconstructions
    {
        boost::filesystem::path reconstruction_filename
            = rootNode.get<std::string>("world_reference");
        if (reconstruction_filename.is_relative())
            reconstruction_filename
                = boost::filesystem::path(filePath).parent_path() / reconstruction_filename;
        visual_marker_mapping::CameraModel cam_model;
        std::map<int, visual_marker_mapping::Camera> reconstructed_cameras;
        visual_marker_mapping::parseReconstructions(
            reconstruction_filename.string(), reconstructed_tags, reconstructed_cameras, cam_model);
        std::cout << "Read reconstructions!" << std::endl;

        reconstructed_map_points = visual_marker_mapping::flattenReconstruction(reconstructed_tags);
    }

    for (const auto& jointNode : rootNode.get_child("hierarchy"))
    {
        JointInfo inf;
        inf.name = jointNode.second.get<std::string>("name");
        inf.type = jointNode.second.get<std::string>("type");
        if (inf.type == "1_dof_joint")
        {
            inf.ticks_to_rad = jointNode.second.get<double>("ticks_to_rad");
            inf.angular_noise_std_dev = jointNode.second.get<double>("angular_noise_std_dev");
        }
        else if (inf.type == "pose")
        {
            inf.ticks_to_rad = 0;
            inf.angular_noise_std_dev = 0;
        }
        else
            throw std::runtime_error("Unknown joint type: " + inf.type);
        inf.parent_to_joint_guess
            = visual_marker_mapping::propertyTree2EigenMatrix<Eigen::Matrix<double, 7, 1> >(
                jointNode.second.get_child("parent_to_joint_pose_guess"));

        inf.parent = jointNode.second.get<std::string>("parent");

        const auto fix = jointNode.second.get_optional<std::string>("fixed");
        if (fix)
            inf.fixed = (*fix == "true");
        else
            inf.fixed = false;

        name_to_joint[inf.name] = joints.size();

        joints.emplace_back(std::move(inf));
    }

    auto locations = rootNode.get_child_optional("locations");
    if (locations)
    {
        for (const auto& locationNode : *locations)
        {
            LocationInfo inf;
            auto pc = locationNode.second.get_child_optional("world_to_location_pose_guess");
            if (pc)
            {
                inf.world_to_location_pose_guess
                    = visual_marker_mapping::propertyTree2EigenMatrix<Eigen::Matrix<double, 7, 1> >(
                        *pc);
            }
            else
            {
                inf.world_to_location_pose_guess << 0, 0, 0, 1, 0, 0, 0;
            }
            const auto fix = locationNode.second.get_optional<std::string>("fixed");
            if (fix)
                inf.fixed = (*fix == "true");
            else
                inf.fixed = false;

            const int id = locationNode.second.get<int>("id");
            optional_location_infos.emplace(id, inf);
        }
    }

    std::map<int, visual_marker_mapping::DetectionResult> detectionResultsByCamId;
    for (const auto& cameraNode : rootNode.get_child("sensors"))
    {
        const std::string sensor_type = cameraNode.second.get<std::string>("sensor_type");
        const int sensor_id = cameraNode.second.get<int>("sensor_id");
        // if (sensor_type!="camera")
        // continue;
        sensor_id_to_parent_joint.emplace(
            sensor_id, cameraNode.second.get<std::string>("parent_joint"));
        if (sensor_type == "camera")
        {
            const int camera_id = sensor_id;

            boost::filesystem::path camera_path = cameraNode.second.get<std::string>("camera_path");
            if (camera_path.is_relative())
                camera_path = boost::filesystem::path(filePath).parent_path() / camera_path;

            // Read cam intrinsics

            const std::string cam_intrin_filename
                = (camera_path / "camera_intrinsics.json").string();

            std::cout << "Trying to load camera intrinsics for camera " << camera_id << " from "
                      << cam_intrin_filename << " ..." << std::endl;

            boost::property_tree::ptree cameraTree;
            boost::property_tree::json_parser::read_json(cam_intrin_filename, cameraTree);
            cameraModelById.emplace(
                camera_id, visual_marker_mapping::propertyTreeToCameraModel(cameraTree));

            // Read marker detections

            const std::string marker_det_filename
                = (camera_path / "marker_detections.json").string();

            std::cout << "Trying to load marker detections for camera " << camera_id << " from "
                      << marker_det_filename << " ..." << std::endl;
            detectionResultsByCamId.emplace(
                camera_id, visual_marker_mapping::readDetectionResult(marker_det_filename));
            std::cout << "Read marker detections for camera " << camera_id << "!" << std::endl;
        }
        else if (sensor_type == "laser_3d")
        {
            laser_sensor_ids.push_back(sensor_id);
        }
        sensor_id_to_type.emplace(sensor_id, sensor_type);
    }

    //	std::ofstream conv("conv.txt");
    for (const auto& calib_frame_node : rootNode.get_child("calibration_frames"))
    {
        CalibrationFrame calib_frame;

        calib_frame.location_id = calib_frame_node.second.get<int>("location_id");

        for (const auto& id_to_cam_model : cameraModelById)
        {
            const int camera_id = id_to_cam_model.first;
            char buffer[256];
            sprintf(buffer, "camera_image_path_%d", camera_id);
            boost::filesystem::path image_path = calib_frame_node.second.get<std::string>(buffer);
            if (image_path.is_relative())
                image_path = boost::filesystem::path(filePath).parent_path() / image_path;

            // find marker observations for this calibration frame
            int detectedImageId = -1;
            for (size_t j = 0; j < detectionResultsByCamId[camera_id].images.size(); j++)
            {
                // this is a hack and should really be ==...
                const boost::filesystem::path df(
                    detectionResultsByCamId[camera_id].images[j].filename);
                //				std::cout << image_path.string() << " "
                //                          << df.string()
                //                          << std::endl;
                if (image_path.filename() == df.filename())
                {
                    detectedImageId = static_cast<int>(j);
                    break;
                }
            }
            if (detectedImageId == -1)
                std::cout << "Error: Did not find target detections for " << image_path.string()
                          << std::endl;
            for (const auto& tagObs : detectionResultsByCamId[camera_id].tagObservations)
            {
                if (tagObs.imageId != detectedImageId)
                    continue;

                for (size_t c = 0; c < 4; c++)
                {
                    const int point_id = tagObs.tagId * 4 + c;
                    calib_frame.cam_id_to_observations[camera_id][point_id] = tagObs.corners[c];
                }
            }
        }

        for (int sensor_id : laser_sensor_ids)
        {
            char buffer[256];
            sprintf(buffer, "scan_file_path_%d", sensor_id);
            boost::filesystem::path scan_path = calib_frame_node.second.get<std::string>(buffer);
            if (scan_path.is_relative())
                scan_path = boost::filesystem::path(filePath).parent_path() / scan_path;

            auto scan = std::make_shared<Scan3D>(loadScan(scan_path.string()));
            if (laser_noise_std_dev > 0.0)
                addNoiseToScan(*scan, laser_noise_std_dev);
            calib_frame.sensor_id_to_laser_scan_3d.emplace(sensor_id, std::move(scan));
            // std::cout << "Loaded: " << scan_path.string() << std::endl;
        }

        // ptuInfo.image_path = image_path.string();
        // ptuInfo.camera_id = ptuPoseNode.second.get<int>("camera_id");

        for (size_t j = 0; j < joints.size(); j++)
        {
            if (joints[j].type == "1_dof_joint")
            {
                char buffer[256];
                sprintf(buffer, "%s_ticks", joints[j].name.c_str());
                const int ticks = calib_frame_node.second.get<int>(buffer);
                calib_frame.joint_config.push_back(ticks);
            }
            else
                calib_frame.joint_config.push_back(0);
        }

//		if (ptuInfo.camera_id==0)
//		{
//			conv << "{\n";
//			conv << "\"location_id\": 0," << std::endl;
//			conv << "\"camera_image_path_0\": \"" << origim.string() << "\"," << std::endl;
//			std::string tmp=origim.string();
//			tmp[3]='1';
//			conv << "\"camera_image_path_1\": \"" << tmp << "\"," << std::endl;
//			conv << "\"joint_ticks_0\": \"" << ptuPoseNode.second.get<int>("joint_0.ticks") << "\","
//<<
// std::endl;
//			conv << "\"joint_ticks_1\": \"" << ptuPoseNode.second.get<int>("joint_1.ticks") << "\","
//<<
// std::endl;
//			conv << "},\n";
//		}

// ptuInfo.cameraIdToObservations

#if 0
        // find marker observations for this calibration frame
        int detectedImageId = -1;
        for (size_t j = 0; j < detectionResultsByCamId[ptuInfo.camera_id].images.size(); j++)
        {
            // this is a hack and should really be ==...
            if (image_path.filename()
                == detectionResultsByCamId[ptuInfo.camera_id].images[j].filename)
            {
                detectedImageId = static_cast<int>(j);
                break;
            }
        }
        for (const auto& tagObs : detectionResultsByCamId[ptuInfo.camera_id].tagObservations)
        {
            if (tagObs.imageId != detectedImageId)
                continue;
            ptuInfo.marker_observations.push_back(tagObs);
            ptuInfo.marker_observations.back().imageId = -1;
        }
#endif

        calib_frames.emplace_back(std::move(calib_frame));
    }
}
//----------------------------------------------------------------------------
}
