// STL
#include <iostream>
#include <string>

// PCL
#include <pcl/common/io.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::cout;
using std::endl;
using std::string;

/* This examples shows how to estimate the SIFT points based on the
 * z gradient of the 3D points than using the Intensity gradient as
 * usually used for SIFT keypoint estimation.
 */

namespace pcl {
template <>
struct SIFTKeypointFieldSelector<PointXYZ> {
  inline float operator()(const PointXYZ &p) const { return p.z; }
};
}  // namespace pcl

/*
 Extract SIFT features from given cloud points

 Args
 - cloud_xyz: cloud points to be extracted

 Return
 - cloud_temp: extracted SIFT cloud points
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr extract_sift(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, float min_scale,
    int n_octaves, int n_scales_per_octave, float min_contrast) {
  // Parameters for sift computation
  // the standard deviation of the smallest scale in the scale space
  // const float min_scale = 0.2f; // the number of octaves (i.e. doublings of
  // scale) to compute const int n_octaves = 10;
  // // the number of scales to compute within each octave
  // const int n_scales_per_octave = 8;

  pcl::console::TicToc time;
  time.tic();
  // Estimate the sift interest points using z values from xyz as the Intensity
  // variants
  pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud_xyz);
  sift.compute(result);
  cout << endl;
  cout << "min_scale = " << min_scale << endl;
  cout << "n_octaves = " << n_octaves << endl;
  cout << "n_scales_per_octave = " << n_scales_per_octave << endl;
  cout << "min_contrast = " << min_contrast << endl;
  cout << "# of SIFT points in the result are " << result.points.size() << endl;

  // Copying the pointwithscale to pointxyz so as visualize the cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(
      new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *cloud_temp);

  return cloud_temp;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    PCL_ERROR(
        "[FORMAT] ./pcl-sift [input file path] [save file path] [min_scale] "
        "[n_octaves] [n_scales_per_octave] [min_contrast]\n");
    return -1;
  }

  string file_in = argv[1];
  string file_out = argv[2];
  float min_scale = strtof(argv[3], 0);
  int n_octaves = strtof(argv[4], 0);
  int n_scales_per_octave = strtof(argv[5], 0);
  float min_contrast = strtof(argv[6], 0);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile<pcl::PointXYZ>(file_in, *cloud_xyz);
  // pcl::io::loadOBJFile<pcl::PointXYZ>(file_in, *cloud_xyz);
  auto cloud_temp = extract_sift(cloud_xyz, min_scale, n_octaves,
                                 n_scales_per_octave, min_contrast);
  pcl::io::savePLYFile(file_out, *cloud_temp, false);

  return 0;
}
