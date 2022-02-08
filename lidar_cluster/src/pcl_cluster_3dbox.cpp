#include <ros/ros.h>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "std_msgs/String.h"

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/TimeReference.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/console/time.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

double MINIMUM_RANGE = 3;
double MAXIMUM_RANGE = 30;
bool isBusy;

std_msgs::Time bag_time;
//visualization_msgs::Marker Cube;
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> ClustersVec;

ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub;
ros::Publisher centroid_pub;
ros::Publisher cube_pub;
ros::Publisher centroid_marker_array_pub;
ros::Publisher Time_pub;




pcl::PointCloud<pcl::PointXYZI>::Ptr VoxelGrid_Filter(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud)
{
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    vg.setInputCloud (input_cloud);
    vg.setLeafSize (0.1, 0.1, 0.1);
    vg.filter (*cloud_filtered);
    return cloud_filtered;
}

// 濾掉最遠的點跟車上最近的pointcloud
void removePointCloud(const pcl::PointCloud<pcl::PointXYZI> &cloud_in, pcl::PointCloud<pcl::PointXYZI> &cloud_out)
{
    //cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());

    size_t j = 0;
    for (size_t i=0; i<cloud_in.points.size(); i++)
    {
        double dis = cloud_in.points[i].x*cloud_in.points[i].x + cloud_in.points[i].y*cloud_in.points[i].y;
        double right_half = cloud_in.points[i].x+cloud_in.points[i].y;  // filter out right half
        if (dis>MAXIMUM_RANGE*MAXIMUM_RANGE || dis<MINIMUM_RANGE*MINIMUM_RANGE)  continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }

    // resize because original is as big as cloud in
    cloud_out.points.resize(j);

    // cloud_out.height = 1;
    // cloud_out.width = static_cast<uint32_t>(j);
    // cloud_out.is_dense = true;
}

void ground_extraction(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud)
{
    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZI> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.3);

    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (input_cloud);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0) {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud (input_cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>);
    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *output_cloud = *cloud_f;

    sensor_msgs::PointCloud2 filtered_cloud;
    pcl::toROSMsg(*cloud_plane, filtered_cloud);
    filtered_cloud.header.frame_id = "velodyne";
    plane_filtered_pub.publish(filtered_cloud);
}

void cluster_box(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud, visualization_msgs::Marker &markerx, visualization_msgs::Marker &markery, visualization_msgs::Marker &markerz, visualization_msgs::Marker &cube)
{
    pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
    feature_extractor.setInputCloud (input_cloud);
    feature_extractor.compute ();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZI min_point_AABB;
    pcl::PointXYZI max_point_AABB;
    pcl::PointXYZI min_point_OBB;
    pcl::PointXYZI max_point_OBB;
    pcl::PointXYZI position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia (moment_of_inertia);
    feature_extractor.getEccentricity (eccentricity);
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter (mass_center);
    /*
    pcl::PointXYZ *point;
    point->x = mass_center(0);
    point->y = mass_center(1);
    point->z = mass_center(2);
    */
    geometry_msgs::Point ps;
    ps.x = mass_center(0) ;
    ps.y = mass_center(1) ;
    ps.z = mass_center(2) ;
    geometry_msgs::Point px;
    px.x = major_vector(0) + mass_center(0);
    px.y = major_vector(1) + mass_center(1);
    px.z = major_vector(2) + mass_center(2);
    geometry_msgs::Point py;
    py.x = middle_vector(0) + mass_center(0);
    py.y = middle_vector(1) + mass_center(1);
    py.z = middle_vector(2) + mass_center(2);
    geometry_msgs::Point pz;
    pz.x = minor_vector(0) + mass_center(0);
    pz.y = minor_vector(1) + mass_center(1);
    pz.z = minor_vector(2) + mass_center(2);
    markerx.points.push_back(ps);
    markerx.points.push_back(px);
    markery.points.push_back(ps);
    markery.points.push_back(py);
    markerz.points.push_back(ps);
    markerz.points.push_back(pz);

    Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Vector3f dis (max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z);
    Eigen::Quaternionf quat (rotational_matrix_OBB);
    Eigen::Vector3f corner = quat*(dis/2);

    Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
    Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
    Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
    Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
    Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
    Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
    Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
    Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
    p1 = rotational_matrix_OBB * p1 + position;
    p2 = rotational_matrix_OBB * p2 + position;
    p3 = rotational_matrix_OBB * p3 + position;
    p4 = rotational_matrix_OBB * p4 + position;
    p5 = rotational_matrix_OBB * p5 + position;
    p6 = rotational_matrix_OBB * p6 + position;
    p7 = rotational_matrix_OBB * p7 + position;
    p8 = rotational_matrix_OBB * p8 + position;

    geometry_msgs::Point e1;
    e1.x = p1 (0);
    e1.y = p1 (1);
    e1.z = p1 (2);
    geometry_msgs::Point e2;
    e2.x = p2 (0);
    e2.y = p2 (1);
    e2.z = p2 (2);
    geometry_msgs::Point e3;
    e3.x = p3 (0);
    e3.y = p3 (1);
    e3.z = p3 (2);
    geometry_msgs::Point e4;
    e4.x = p4 (0);
    e4.y = p4 (1);
    e4.z = p4 (2);
    geometry_msgs::Point e5;
    e5.x = p5 (0);
    e5.y = p5 (1);
    e5.z = p5 (2);
    geometry_msgs::Point e6;
    e6.x = p6 (0);
    e6.y = p6 (1);
    e6.z = p6 (2);
    geometry_msgs::Point e7;
    e7.x = p7 (0);
    e7.y = p7 (1);
    e7.z = p7 (2);
    geometry_msgs::Point e8;
    e8.x = p8 (0);
    e8.y = p8 (1);
    e8.z = p8 (2);

    cube.points.push_back(e1);
    cube.points.push_back(e2);
    cube.points.push_back(e1);
    cube.points.push_back(e4);
    cube.points.push_back(e1);
    cube.points.push_back(e5);
    cube.points.push_back(e5);
    cube.points.push_back(e6);
    cube.points.push_back(e5);
    cube.points.push_back(e8);
    cube.points.push_back(e2);
    cube.points.push_back(e6);
    cube.points.push_back(e6);
    cube.points.push_back(e7);
    cube.points.push_back(e7);
    cube.points.push_back(e8);
    cube.points.push_back(e2);
    cube.points.push_back(e3);
    cube.points.push_back(e4);
    cube.points.push_back(e8);
    cube.points.push_back(e3);
    cube.points.push_back(e4);
    cube.points.push_back(e3);
    cube.points.push_back(e7);

    //return point;
}


void publish_centroid_marker(pcl::PointCloud<pcl::PointXYZ> centroids)
{
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/velodyne";
    marker.ns = "centroid";

    // clear all markers first, suppose at most 200 markers
    marker.action = visualization_msgs::Marker::DELETE;
    for (int i=0; i<200; i++) {
        marker.id = i;
        marker_array.markers.push_back(marker);
    }
    centroid_marker_array_pub.publish(marker_array);


    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    marker.scale.z = 1.0;
    marker.color.b = 1.0;
    marker.color.g = 1.0;
    marker.color.r = 1.0;
    marker.color.a = 1.0;


    for (int i=0; i<centroids.points.size(); i++) {
        marker.id = i;
        geometry_msgs::Pose pose;

        pose.position.x = centroids.points[i].x;
        pose.position.y = centroids.points[i].y;
        pose.position.z = centroids.points[i].z;

        char display[50];
        sprintf(display, "%.2f %.2f %.2f", centroids.points[i].x, centroids.points[i].y, centroids.points[i].z);

        marker.text = display;
        marker.pose = pose;
        marker_array.markers.push_back(marker);
    }

    centroid_marker_array_pub.publish(marker_array);
}

void publish_centroid(pcl::PointCloud<pcl::PointXYZ> centroids)
{
    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(centroids, cluster_cloud);
    cluster_cloud.header.frame_id = "velodyne";
    centroid_pub.publish(cluster_cloud);
    Time_pub.publish(bag_time);
}

void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    bag_time.data =  msg->header.stamp;
    
    if(isBusy) return;
    isBusy = true;
    ClustersVec.clear();

    // === Msg to PointCloud === //

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    fromROSMsg(*msg, *cloud);
    //pcl::console::TicToc tt;
    //tt.tic();

    // === Remove outlier point === //

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_remove(new pcl::PointCloud<pcl::PointXYZI>);
    removePointCloud(*cloud, *cloud_remove);
    
    /*
    // === Downsample === //

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    cloud_filtered = VoxelGrid_Filter(cloud_remove);
    */

    // === Ground segmentation === //

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_remove_ground_extract(new pcl::PointCloud<pcl::PointXYZI>);
    ground_extraction(cloud_remove, cloud_remove_ground_extract);
    //std::cout << "filter time(ms): " << tt.toc() << std::endl;
    //tt.tic();

    
    // === Get cluster indices === //

    // creating the KdTree object for the search method of the extraction 
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_remove_ground_extract);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.4);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(2000); 
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_remove_ground_extract);
    ec.extract(cluster_indices); // [[index0,1,2], [5,3,4]......]

    // === Compute Cluster Centroid === //

    int j = 50; // cluster intensity

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr centers (new pcl::PointCloud<pcl::PointXYZ>);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) 
    {
        // === Extract clusters and save as a single point cloud === //
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster_filtered (new pcl::PointCloud<pcl::PointXYZI>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
            cloud_remove_ground_extract->points[*pit].intensity = j;
            cloud_cluster->points.push_back(cloud_remove_ground_extract->points[*pit]); //*s
        }

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // ================
        cloud_cluster_filtered = VoxelGrid_Filter(cloud_cluster);

        pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
        feature_extractor.setInputCloud (cloud_cluster);
        feature_extractor.compute ();

        pcl::PointXYZI min_point_AABB;
        pcl::PointXYZI max_point_AABB;
        Eigen::Vector3f mass_center;

        feature_extractor.getAABB (min_point_AABB, max_point_AABB);
        feature_extractor.getMassCenter (mass_center);

        if(max_point_AABB.z - min_point_AABB.z > 5.0f ||  max_point_AABB.z - min_point_AABB.z < 0.5f 
        || mass_center[2] < -2.5 || mass_center[2] > 1)
            continue;
        
        ClustersVec.push_back(cloud_cluster_filtered);

        j += 2;

        // === Compute centroid === //
        //Eigen::Vector4f centroid;
        //pcl::compute3DCentroid(*cloud_cluster, centroid);
        float x = mass_center[0];
        float y = mass_center[1];
        float z = mass_center[2];
        pcl::PointXYZ point{x,y,z};
        centers->push_back(point);
    }

    publish_centroid_marker(*centers); // display centroid pos on screen
    publish_centroid(*centers);  // publish centroid to main.cpp


    //CloudClusters = cloud_clusters;
    //Cube = cube;
    //tt.tic();
    //std::cout << "final time(ms): " << tt.toc() << " ----------------------" << std::endl;
}

void complete_callback(const std_msgs::Int32MultiArray::ConstPtr &msg)
{
    
    visualization_msgs::Marker cube;
    visualization_msgs::Marker markerx;
    visualization_msgs::Marker markery;
    visualization_msgs::Marker markerz;
    markerx.header.frame_id = markery.header.frame_id = markerz.header.frame_id = cube.header.frame_id = "/velodyne";
    markerx.header.stamp = markery.header.stamp = markerz.header.stamp = cube.header.stamp = ros::Time::now();
    markerx.type = markery.type = markerz.type = cube.type = visualization_msgs::Marker::LINE_LIST;
    markerx.scale.x = markery.scale.x = markerz.scale.x = cube.scale.x = 0.05;
    markerx.scale.y = markery.scale.y = markerz.scale.y = cube.scale.y = 0.05;
    markerx.scale.z = markery.scale.z = markerz.scale.z = cube.scale.z = 0.05;
    markerx.color.a = markery.color.a = markerz.color.a = cube.color.a = 1.0;
    markerx.color.b = 255; markerx.color.g = 250; markerx.color.r = 250;
    markery.color.b = 255; markery.color.g = 250; markery.color.r = 250;
    markerz.color.b = 255; markerz.color.g = 250; markerz.color.r = 250;
    cube.color.b = 1.0; cube.color.g = 1.0; cube.color.r = 1.0;
    



    pcl::PointCloud<pcl::PointXYZI>::Ptr CloudClusters (new pcl::PointCloud<pcl::PointXYZI>);
    CloudClusters->clear();

    for(int i=0; i<msg->data.size(); i++)
    {
        *CloudClusters += *(ClustersVec[msg->data[i]]);
        cluster_box(ClustersVec[msg->data[i]],markerx,markery,markerz,cube);
        //std::cout << msg->data[i] << std::endl;
    }

    cube_pub.publish(cube);

    sensor_msgs::PointCloud2 cluster_cloud;
    pcl::toROSMsg(*CloudClusters, cluster_cloud);
    cluster_cloud.header.frame_id = "velodyne";
    cluster_pub.publish(cluster_cloud);

    isBusy = false;
}



int main(int argc, char **argv) {
    ros::init(argc, argv, "node");
    ros::NodeHandle nh;

    plane_filtered_pub          = nh.advertise<sensor_msgs::PointCloud2 >("plane_filtered_pub_points", 10);
    cluster_pub                 = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 10);
    centroid_pub                = nh.advertise<sensor_msgs::PointCloud2 >("centroid", 10);
    cube_pub                    = nh.advertise<visualization_msgs::Marker>("bounding_boxs", 10);
    centroid_marker_array_pub   = nh.advertise<visualization_msgs::MarkerArray>("/centroid_marker_array", 10);
    Time_pub                    = nh.advertise<std_msgs::Time>("/bag_time", 1);

    ros::Subscriber lidar_sub       = nh.subscribe("points_raw", 10, lidar_callback);
    ros::Subscriber complete_sub    = nh.subscribe("complete_check", 1, complete_callback);

    ros::spin();
    return 0;
}
