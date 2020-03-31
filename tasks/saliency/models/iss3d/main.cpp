#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <time.h>
#include <pcl/common/io.h>
#include <iostream>
#include <pcl/keypoints/iss_3d.h>
#include <cstdlib>
#include <pcl/io/io.h>
using namespace std;
using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int
main(int argc, char** argv)
{
    string file_in = argv[1];
    string file_out = argv[2];

    PointCloud::Ptr cloud_src_o(new PointCloud);
    pcl::io::loadPLYFile(file_in, *cloud_src_o);
    cout << "/////////////////////////////////////////////////" << endl;
    cout << "original point cloud:"<<cloud_src_o->size() << endl;


    clock_t start = clock();

    //std::vector<int> indices_src;
    //pcl::removeNaNFromPointCloud(*cloud_src_o, *cloud_src_o, indices_src);
    //std::cout << "remove *cloud_src_o nan" << cloud_src_o->size()<<endl;

    //std::vector<int> indices_tgt;
    //pcl::removeNaNFromPointCloud(*cloud_tgt_o, *cloud_tgt_o, indices_tgt);
    //std::cout << "remove *cloud_tgt_o nan" << cloud_tgt_o->size()<<endl;



    //iss keypoint
    PointCloud::Ptr cloud_src_iss(new PointCloud);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoint(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_1(new pcl::search::KdTree<pcl::PointXYZ>());

    double model_resolution = 0.015;

    //≤Œ ˝…Ë÷√
    iss_det.setSearchMethod(tree_1);
    iss_det.setSalientRadius(6 * model_resolution);
    iss_det.setNonMaxRadius(4 * model_resolution);
    iss_det.setNormalRadius(4 * model_resolution);
    iss_det.setBorderRadius(0);
    iss_det.setThreshold21(0.995);
    iss_det.setThreshold32(0.995);
    iss_det.setMinNeighbors(5);
    iss_det.setNumberOfThreads(4);
    iss_det.setInputCloud(cloud_src_o);
    iss_det.compute(*cloud_src_iss);


    clock_t end = clock();
    cout << "time:" << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "keypoint number:" << cloud_src_iss->size() << endl;

    PointCloud::Ptr cloud_src(new PointCloud);
    pcl::copyPointCloud(*cloud_src_iss, *cloud_src);

    pcl::io::savePLYFile(file_out, *cloud_src_iss, false);

    return (0);
}
