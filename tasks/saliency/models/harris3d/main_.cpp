#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/harris_3d.h>
#include <cstdlib>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<float> compute(py::array_t<float> pc, double radius, double threshold)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZI>);

	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
    for (int i = 0; i < pc.shape(0); i++){
        std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
        pc_ptr += 3;
    }
    std::cout << "input cloud size: " << cloud->size() << std::endl;

    //for(int i = 0; i < 5; i ++){
    //    std::cout << cloud->points[i].data[0] << ", " << cloud->points[i].data[1] << ", " << cloud->points[i].data[2] << std::endl;
    //}
    pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI,pcl::Normal> harris;
    harris.setInputCloud(cloud);
    std::cout<<"input successful"<<std::endl;
    harris.setNonMaxSupression(true);
    harris.setRadius(radius);
    harris.setThreshold(threshold);
    std::cout<<"parameter set successful"<<std::endl;

    cloud_out->height=1;
    cloud_out->width =100; //unordered
    cloud_out->resize(cloud_out->height*cloud->width);
    cloud_out->clear();
    harris.compute(*cloud_out);
    int size = cloud_out->size();
    std::cout << "harris kp size: " << size << std::endl;
    pcl::PointIndicesConstPtr keypoints_indices = harris.getKeypointsIndices();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_harris (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_harris->height=1;
    cloud_harris->width =100;
    cloud_harris->resize(cloud_out->height*cloud->width);
    cloud_harris->clear();

    pcl::PointXYZ point;

    for (int i = 0; i<size;i++)
    {
        point.x = cloud_out->at(i).x;
        point.y = cloud_out->at(i).y;
        point.z = cloud_out->at(i).z;
        cloud_harris->push_back(point);
    }

    //pcl::io::savePLYFile("file_out.ply", *cloud_out, false);
    auto result = py::array_t<float>(cloud_out->size());// * 3);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;

    for (int i = 0; i < cloud_out->size(); ++i){
        ptr[i] = keypoints_indices->indices[i];
    }

    //   for (int i = 0; i < cloud_out->size(); ++i)
    //{
	//	std::copy(&cloud_out->points[i].data[0], &cloud_out->points[i].data[3], &ptr[i * 3]);
    //}
    return result;
}

PYBIND11_MODULE(harris3d, m){
    m.def("compute",  &compute, py::arg("pc"), py::arg("radius")=0.05, py::arg("threshold")=0.05);
}
