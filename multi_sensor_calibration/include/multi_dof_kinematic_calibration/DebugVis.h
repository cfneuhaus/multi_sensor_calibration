#ifndef MULTI_DOF_KINEMATIC_CALIBRATION_DEBUGVIS_H
#define MULTI_DOF_KINEMATIC_CALIBRATION_DEBUGVIS_H

#include <string>
#include <exception>
#include <Eigen/Core>
#include <vector>
#include <fstream>

class DebugOutput
{
public:
    DebugOutput(const std::string& filename)
        : filename(filename)
    {
    }
    DebugOutput(const DebugOutput&) = delete;
    void operator=(const DebugOutput&) = delete;
    ~DebugOutput()
    {
        std::ofstream f(filename);
        if (!f.good())
            throw std::runtime_error("Cant write " + filename);
        f << "{ \"reconstructed_cameras\" : [";

        for (size_t i = 0; i < poses.size(); i++)
        {
            const auto& p = poses[i];
            f << "{";
            f << "\"caption\":\"" << pose_captions[i] << "\",";
            f << "\"rotation\":[";
            f << p(3) << "," << p(4) << "," << p(5) << "," << p(6) << "],";
            f << "\"translation\":[";
            f << p(0) << "," << p(1) << "," << p(2) << "]";
            f << "}";
            if (i + 1 < poses.size())
                f << ",\n" << std::endl;
        }
        f << "],\n";

        f << "\"points\" : [";
        for (size_t i = 0; i < points.size(); i++)
        {
            f << "[";
            f << points[i](0) << ",";
            f << points[i](1) << ",";
            f << points[i](2) << "]";

            if (i + 1 < points.size())
                f << ",";
        }
        f << "],";

        f << "\"lines\" : [";
        for (size_t i = 0; i < line_is_vector.size(); i++)
        {
            f << "{";
            f << "\"from\":[";
            f << lines[2 * i](0) << ",";
            f << lines[2 * i](1) << ",";
            f << lines[2 * i](2) << "],";
            f << "\"to\":[";
            f << lines[2 * i + 1](0) << ",";
            f << lines[2 * i + 1](1) << ",";
            f << lines[2 * i + 1](2) << "],";
            f << "\"is_vector\":" << (line_is_vector[i] ? "true" : "false");
            f << "}";

            if (i + 1 < line_is_vector.size())
                f << ",";
        }
        f << "]";


        f << "}";
    }

    void addPose(const Eigen::Matrix<double, 7, 1>& pose, const std::string& caption = "")
    {
        poses.push_back(pose);
        pose_captions.push_back(caption);
    }
    void addLine(const Eigen::Vector3d& a, const Eigen::Vector3d& b, bool is_vector = false)
    {
        lines.push_back(a);
        lines.push_back(b);
        line_is_vector.push_back(is_vector);
    }
    void addPoint(const Eigen::Vector3d& point) { points.push_back(point); }

private:
    std::vector<Eigen::Matrix<double, 7, 1> > poses;
    std::vector<std::string> pose_captions;
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> lines;
    std::vector<bool> line_is_vector;
    std::string filename;
};

#endif
