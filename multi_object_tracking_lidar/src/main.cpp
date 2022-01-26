#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include "std_msgs/String.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <limits>
#include <utility>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <darknet_ros_msgs/BoundingBoxes.h>

// Hungarian
#include "kf_tracker/Hungarian.h"
#include "kf_tracker/getMatch.h"

std::string CSV_OUTPUT_ADDR = "/home/an/test_ws/src/Tracking_01/path_test.csv";

ros::Publisher id_marker_array_pub;
ros::Publisher Complete_pub;
ros::Publisher image_pub;

int WIDTH = 1280, HEIGHT = 720;
Eigen::Matrix4f l_t_c;
cv::Mat INTRINSIC_MAT(3, 3, cv::DataType<double>::type); // Intrinsics
cv::Mat DIST_COEFFS(5, 1, cv::DataType<double>::type); // Distortion vector
cv::Mat ROTATE_MAT_Rodrigues(3, 1, cv::DataType<double>::type);
cv::Mat TRANSLATE_MAT(3, 1, cv::DataType<double>::type);
std::vector<cv::Point3f> rVec;
std::vector<cv::Point3f> tVec;

bool isBusy;
std::vector<int> trackingList_idx;
std_msgs::Time bag_time;
pcl::PointCloud<pcl::PointXYZ>::Ptr Centroid(new pcl::PointCloud<pcl::PointXYZ>);

class YOLO_BoundingBox;

std::vector<YOLO_BoundingBox> YOLO_BB;



void initializeGlobalParams()
{
    cv::Mat ROTATE_MAT(3, 3, cv::DataType<double>::type);

    l_t_c << 0.84592974185943604, 0.53328412771224976, -0.0033089336939156055, 0.092240132391452789,
            0.045996580272912979, -0.079141519963741302, -0.99580162763595581, -0.35709697008132935,
            -0.53130710124969482, 0.84222602844238281, -0.091477409005165100, -0.16055910289287567,
            0, 0, 0, 1;

    ROTATE_MAT.at<double>(0,0)=0.84592974185943604;
    ROTATE_MAT.at<double>(0,1)=0.53328412771224976;
    ROTATE_MAT.at<double>(0,2)=-0.0033089336939156055;
    ROTATE_MAT.at<double>(1,0)=0.045996580272912979;
    ROTATE_MAT.at<double>(1,1)=-0.079141519963741302;
    ROTATE_MAT.at<double>(1,2)=-0.99580162763595581;
    ROTATE_MAT.at<double>(2,0)=-0.53130710124969482;
    ROTATE_MAT.at<double>(2,1)=0.84222602844238281;
    ROTATE_MAT.at<double>(2,2)=-0.091477409005165100;

    cv::Rodrigues(ROTATE_MAT, ROTATE_MAT_Rodrigues);

    TRANSLATE_MAT.at<double>(0,0) = 0.092240132391452789;
    TRANSLATE_MAT.at<double>(1,0) = -0.35709697008132935;
    TRANSLATE_MAT.at<double>(2,0) = -0.16055910289287567;

    INTRINSIC_MAT.at<double>(0, 0) = 698.939;
    INTRINSIC_MAT.at<double>(1, 0) = 0;
    INTRINSIC_MAT.at<double>(2, 0) = 0;

    INTRINSIC_MAT.at<double>(0, 1) = 0;
    INTRINSIC_MAT.at<double>(1, 1) = 698.939;
    INTRINSIC_MAT.at<double>(2, 1) = 0;

    INTRINSIC_MAT.at<double>(0, 2) = 641.868;
    INTRINSIC_MAT.at<double>(1, 2) = 385.788;
    INTRINSIC_MAT.at<double>(2, 2) = 1.0;

    DIST_COEFFS.at<double>(0) = -0.171466;
    DIST_COEFFS.at<double>(1) = 0.0246144;
    DIST_COEFFS.at<double>(2) = 0;
    DIST_COEFFS.at<double>(3) = 0;
    DIST_COEFFS.at<double>(4) = 0;

    rVec.push_back(cv::Point3f(0,0,0));
    tVec.push_back(cv::Point3f(0,0,0));
}


class YOLO_BoundingBox
{
    public:
    YOLO_BoundingBox(std::string _label, float _xmin, float _xmax, float _ymin, float _ymax)
    {
        this->label = _label;
        this->xmin = _xmin;
        this->xmax = _xmax;
        this->ymin = _ymin;
        this->ymax = _ymax;
        this->center = cv::Point2f( (_xmin+_xmax)/2, (_ymin+_ymax)/2 );
    }

    bool isMatched(cv::Point2f point)
    {
        if( !(this->label == "car" || this->label == "person") )
        {
            //std::cout << this->label << std::endl;
            return false;
        }
        if( point.x < this->xmin || point.x > this->xmax ||
            point.y < this->ymin || point.y > this->ymax   )
        {
            //std::cout << "========== outside of box ============" << std::endl;
           // || (cv::norm(cv::Mat(this->center), cv::Mat(point)) > 200) )
            return false;
        }

        //std::cout << "matched!!!!!!!!!!!!!!!" << std::endl;

        return true;
    }

    public:
    std::string label;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    cv::Point2f center;
};


class Track
{
    public:
        Track(int track_id, float x, float y, float z, std::string l)
        {
            id = track_id;
            hits = 0;
            no_losses = 0;
            fst_det = 0;
            label = l;
            fish_trace.clear();

            // kalman Filter init
            KF = cv::KalmanFilter(stateDim, measDim, ctrlDim, CV_32F);
            KF.transitionMatrix = (cv::Mat_<float>(6,6) <<  1, 0, 0, 1, 0, 0,
 													        0, 1, 0, 0, 1, 0,
                                                            0, 0, 1, 0, 0, 1,
                                                            0, 0, 0, 1, 0, 0,
                                                            0, 0, 0, 0, 1, 0,
                                                            0, 0, 0, 0, 0, 1 );
            cv::setIdentity(KF.measurementMatrix);
            cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(sigmaP));
            cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(sigmaQ));
            cv::setIdentity(KF.errorCovPost, cv::Scalar::all(0.1));

            KF.statePost.at<float>(0) = x;
            KF.statePost.at<float>(1) = y;
            KF.statePost.at<float>(2) = z;
            KF.statePost.at<float>(3) = 0.4;  // initial v_x
            KF.statePost.at<float>(4) = -0.5;  // initial v_y
            KF.statePost.at<float>(5) = 0;
        }

    public:
        int id;
        int hits;
        int no_losses;
        int fst_det;
        std::string label;
        std::vector<geometry_msgs::Point> fish_trace;

        // Kalman Filter
        int stateDim = 6;  // [x, y, z, v_x, v_y, v_z]
        int measDim = 3;   // [z_x, z_y, z_z]
        int ctrlDim = 0;
        float sigmaP = 0.001;
        float sigmaQ = 0.00001;
        cv::KalmanFilter KF;
};


class Tracker
{
    public:
        Tracker()
        {
            track_id = 0;
            max_age = 4;
            min_hits = 2;
            track_length = 10;


            tracker_list.clear();
            good_tracker_list.clear();
            unmatched_trackers.clear();
            unmatched_detections.clear();
            matches.clear();

            // 開啟檔案為輸出狀態，若檔案已存在則清除檔案內容重新寫入
            position_file.open(CSV_OUTPUT_ADDR, std::ios::out);
            // 檢查檔案是否成功開啟
            if (!position_file) {
                std::cerr << "Can't open file!\n";
                exit(1);     //在不正常情形下，中斷程式的執行
            }
        }

        double _point_distance(geometry_msgs::Point &a, geometry_msgs::Point &b)
        {
            double dis = sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
            return dis;
        }

        void _assign_detections_to_trackers(std::vector<geometry_msgs::Point> &trackers, std::vector<geometry_msgs::Point> &detections, float max_length)
        {
            //HungarianAlgorithm H;
            vector<vector<double>> dis_mat;
            std::vector<int> matched_idx;

            matched_idx.clear();
            unmatched_trackers.clear();
            unmatched_detections.clear();
            matches.clear();


            // === dis_mat initialization === //

            dis_mat.resize(trackers.size());

            for(int t=0; t<trackers.size(); t++) {
                dis_mat[t].resize(detections.size());

                for(int d=0; d<detections.size(); d++) {
                    dis_mat[t][d] = _point_distance(trackers[t], detections[d]);
                }
            }

            // === Match trackers and detections === //

            if(dis_mat.size() > 0)
                //H.Solve(dis_mat, matched_idx);
                getmatch(dis_mat, matched_idx);

            // === Set unmatched trackers === //

            for(int t=0; t<trackers.size(); t++) {
                if(matched_idx[t] == -1) {
                    unmatched_trackers.push_back(t);
                    //std::cout << "add unmatched_trackers" << std::endl;
                 }
            }

            // === Set unmatched detections === //

            for(int d=0; d<detections.size(); d++) {
                vector<int>::iterator it = std::find(matched_idx.begin(), matched_idx.end(), d);
                if(it == matched_idx.end()) {
                    unmatched_detections.push_back(d);
                    //std::cout << "add unmatched_detections" << std::endl;
                }
            }

            // === Set matched pair === //

            for(int t=0; t<trackers.size(); t++) {
                if(matched_idx[t] == -1) continue;
                // 找到太遠的也不符
                if(dis_mat[t][matched_idx[t]] > (max_length + 0.3*tracker_list[t].no_losses) ) {
                    unmatched_trackers.push_back(t);
                    unmatched_detections.push_back(matched_idx[t]);
                } else {
                    std::pair<int,int> m(t, matched_idx[t]);
                    matches.push_back(m);
                    //std::cout << "has match" << endl;
                }
            }
        }

        void update(std::vector<geometry_msgs::Point> &z_box)
        {
            std::vector<geometry_msgs::Point> x_box;
            std::vector<YOLO_BoundingBox> tmp_YOLO_BB;

            matched_trackers_list.clear();
            unmatched_trackers_list.clear();
            x_box.clear();
            tmp_YOLO_BB = YOLO_BB;

            for(int i=0; i<tracker_list.size(); i++) {
                x_box.push_back(tracker_list[i].fish_trace.back());
            }

            // x_box = track, z_box = detect
            _assign_detections_to_trackers(x_box ,z_box, 3);


            // === Deal with matched detections === //
            // 更新卡爾曼濾波
            if(matches.size() > 0) {
                for(int i=0; i<matches.size(); i++) {

                    int trk_idx = matches[i].first;
                    int det_idx = matches[i].second;

                    // YOLO check
                    geometry_msgs::Point pt = z_box[det_idx];
                    std::vector<cv::Point3f> input;
                    std::vector<cv::Point2f> output;
                    bool flag_isMatched = false;

                    input.push_back(cv::Point3f(pt.x, pt.y, pt.z));
                    cv::projectPoints(input,
                                      ROTATE_MAT_Rodrigues,
                                      TRANSLATE_MAT,
                                      INTRINSIC_MAT,
                                      DIST_COEFFS,
                                      output );
                    int BB;
                    for(BB = 0; BB < tmp_YOLO_BB.size(); BB++)
                    {
                        if(tmp_YOLO_BB[BB].label == tracker_list[trk_idx].label)
                        {
                            flag_isMatched = true;
                            break;
                        }
                    }

                    if(!flag_isMatched){
                        unmatched_trackers.push_back(trk_idx);
                        unmatched_detections.push_back(det_idx);
                        continue;
                    } 
                    // === End YOLO Check === //


                    trackingList_idx.push_back(det_idx);

                    //geometry_msgs::Point pt = z_box[det_idx];
                    Track tmp_trk = tracker_list[trk_idx];

                    if(tmp_trk.id == -1)
                        tmp_trk.id = track_id ++;


                    cv::Mat measurement = cv::Mat::zeros(tmp_trk.measDim, 1, CV_32F);
                    measurement.at<float>(0)= (float)pt.x;
			        measurement.at<float>(1) = (float)pt.y;
                    measurement.at<float>(2) = (float)pt.z;
                    cv::Mat correct_pt = tmp_trk.KF.correct(measurement);

                    geometry_msgs::Point pt_2;
                    pt_2.x = correct_pt.at<float>(0);
                    pt_2.y = correct_pt.at<float>(1);
                    pt_2.z = correct_pt.at<float>(2);

                    //std::cout << "correct speed " << correct_pt.at<float>(2) << " " << correct_pt.at<float>(3) << std::endl;

                    tmp_trk.fish_trace.push_back(pt_2);

                    tmp_trk.hits += 1;
                    tmp_trk.no_losses = 0;

                    tracker_list[trk_idx] = tmp_trk;

                    matched_trackers_list.push_back(tmp_trk);
                }
            }

            // Deal with unmatched detections
            // 創建新track並只predict
            if(unmatched_detections.size() > 0) {
                for(int idx=0; idx<unmatched_detections.size(); idx++) {
                    geometry_msgs::Point pt = z_box[unmatched_detections[idx]];
                    //
                    std::vector<cv::Point3f> input;
                    std::vector<cv::Point2f> output;
                    bool flag_isMatched = false;

                    input.push_back(cv::Point3f(pt.x, pt.y, pt.z));
                    if(1.2*pt.x - pt.y >=0 )
                        continue;
                    cv::projectPoints(input,
                                      ROTATE_MAT_Rodrigues,
                                      TRANSLATE_MAT,
                                      INTRINSIC_MAT,
                                      DIST_COEFFS,
                                      output );
                    //std::cout << output[0].x << " " << output[0].y << " " << endl;
                    int BB;
                    for(BB = 0; BB < tmp_YOLO_BB.size(); BB++)
                    {
                        if(tmp_YOLO_BB[BB].isMatched(output[0]))
                        {
                            flag_isMatched = true;
                            break;
                        }
                    }
                    if(!flag_isMatched) continue;

                    // in YOLO Bounding BOX
                    Track tmp_trk = Track(-1, pt.x, pt.y, pt.z, tmp_YOLO_BB[BB].label);
                    //track_id += 1;

                    //std::cout << pt.x << "    " << pt.y << std::endl;

                    cv::Mat predict_pt = tmp_trk.KF.predict();

                    //std::cout << "create and predict speed " << predict_pt.at<float>(2) << " " << predict_pt.at<float>(3) << std::endl;

                    geometry_msgs::Point pt_2;
                    pt_2.x = predict_pt.at<float>(0);
                    pt_2.y = predict_pt.at<float>(1);
                    pt_2.z = predict_pt.at<float>(2);

                    //std::cout << pt_2.x << "    " << pt_2.y << std::endl;

                    tmp_trk.fish_trace.push_back(pt_2);



                    tracker_list.push_back(tmp_trk);
                }
            }

            // Deal with unmatched tracks
            // 只predict
            if(unmatched_trackers.size() > 0) {
                for(int trk_idx=0; trk_idx<unmatched_trackers.size(); trk_idx++) {
                    Track tmp_trk = tracker_list[unmatched_trackers[trk_idx]];

                    //std::cout << "position before predicting " << tmp_trk.fish_trace.back().x << " " << tmp_trk.fish_trace.back().y << std::endl;

                    // 遺失追蹤
                    tmp_trk.no_losses += 1;
                    // 直接predict下個狀態
                    cv::Mat predict_pt = tmp_trk.KF.predict();

                    //std::cout << "position after predicting " << predict_pt.at<float>(0) << " " << predict_pt.at<float>(1) << std::endl;
                    //std::cout << "only predict speed " << predict_pt.at<float>(2) << " " << predict_pt.at<float>(3) << std::endl;

                    geometry_msgs::Point pt;
                    pt.x = predict_pt.at<float>(0);
                    pt.y = predict_pt.at<float>(1);
                    pt.z = predict_pt.at<float>(2);

                    tmp_trk.fish_trace.push_back(pt);

                    tracker_list[unmatched_trackers[trk_idx]] = tmp_trk;

                    unmatched_trackers_list.push_back(tmp_trk);
                }
            }

            std::vector<Track> temp_tracker_list;
            temp_tracker_list.clear();

            //std::cout << tracker_list.size() << endl;

            for(int i=0; i<tracker_list.size(); i++) {
                Track trk = tracker_list[i];
                if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)) {
                    good_tracker_list.push_back(trk);
                }
                if (trk.no_losses <= max_age) {
                    temp_tracker_list.push_back(trk);
                }
            }

            // update
            tracker_list.clear();
            for(int i=0; i<temp_tracker_list.size(); i++) {
                tracker_list.push_back(temp_tracker_list[i]);
            }

            publish_id();

        }

        void publish_id()
        {
            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker marker;
            marker.header.frame_id = "/velodyne";
            marker.ns = "id";

            // clear all markers first, suppose at most 200 markers
            
            marker.action = visualization_msgs::Marker::DELETE;
            for (int i=0; i<200; i++) {
                marker.id = i;
                marker_array.markers.push_back(marker);
            }
            id_marker_array_pub.publish(marker_array);
            

            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

            marker.scale.z = 1.5;
            marker.color.b = 1.0;
            marker.color.g = 1.0;
            marker.color.r = 1.0;
            marker.color.a = 1.0;

            //std::cout << matched_trackers_list.size() << endl;

            for (int i=0; i<matched_trackers_list.size(); i++) {
                // === Display ID and Label === //
                marker.id = i;
                geometry_msgs::Pose pose;

                pose.position.x = matched_trackers_list[i].fish_trace.back().x;
                pose.position.y = matched_trackers_list[i].fish_trace.back().y;
                pose.position.z = matched_trackers_list[i].fish_trace.back().z;

                /*
                std::cout << matched_trackers_list[i].id << " "
                          << pose.position.x << " "
                          << pose.position.y << " "
                          << pose.position.z<< std::endl;
                */

                char display[20];
                sprintf(display, "%d ", matched_trackers_list[i].id);

                marker.text = display+matched_trackers_list[i].label;
                marker.pose = pose;
                marker_array.markers.push_back(marker);

                // === OUTPUT CSV === //
                std::ostringstream time_stamp, x, y, z;
                time_stamp << bag_time.data;
                x <<  matched_trackers_list[i].fish_trace.back().x;
                y <<  matched_trackers_list[i].fish_trace.back().y;
                z <<  matched_trackers_list[i].fish_trace.back().z;
                std::string time_stamp_str(time_stamp.str());
                std::string x_str(x.str());
                std::string y_str(y.str());
                std::string z_str(z.str());
                std::string concate_str = time_stamp_str + ", "+display+ ", " + x_str + ", " + y_str + ", "+ z_str + "\n";
                std::cout << concate_str;
                position_file << concate_str;
            }

            std::cout << std::endl;

            id_marker_array_pub.publish(marker_array);
        }

    private:
        int track_id;
        int max_age;
        int min_hits;
        int track_length;
        
        std::vector<Track> tracker_list;
        std::vector<Track> good_tracker_list;

        std::vector<Track> matched_trackers_list;
        std::vector<Track> unmatched_trackers_list;
        //std::vector<Track> unmatched_detections_list;

        std::vector<int> unmatched_trackers;
        std::vector<int> unmatched_detections;
        std::vector<std::pair<int,int>> matches;

        std::ofstream position_file;

};
Tracker T;



void test()
{
    std::vector<std::vector<double>> costMatrix = { { 5.0, 10.0, 25.0, 40.0},
										            { 10.0, 5.0, 20.0, 25.0},
                                                    { 30.0, 10.0,10.0, 25.0},
                                                    { 30.0, 20.0, 5.0, 10.0}
                                                    };
    std::cout << costMatrix[1][2] << std::endl;

	HungarianAlgorithm HungAlgo;
	std::vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++)
		std::cout << x << "," << assignment[x] << "\t";

	std::cout << "\ncost: " << cost << std::endl;
}



void centroid_callback(const sensor_msgs::PointCloud2::ConstPtr &input) {

    isBusy = true;
    pcl::PointCloud<pcl::PointXYZ>::Ptr centroids(new pcl::PointCloud<pcl::PointXYZ>);
    fromROSMsg(*input, *centroids);
    Centroid = centroids;

    std::vector<geometry_msgs::Point> temp;
    temp.clear();
    trackingList_idx.clear();

    for(int i=0; i<centroids->size(); i++) {
        geometry_msgs::Point pt;
        pt.x = (*centroids)[i].x;
        pt.y = (*centroids)[i].y;
        pt.z = (*centroids)[i].z;

        temp.push_back(pt);
    }

    T.update(temp);

    std_msgs::Int32MultiArray msg;
    msg.data.clear();

    for(int i=0; i<trackingList_idx.size(); i++){
        msg.data.push_back(trackingList_idx[i]);
        //std::cout << trackingList_idx[i] << std::endl;
    }

    Complete_pub.publish(msg);

    isBusy = false;
}

void cameraImage_callback(const sensor_msgs::Image::ConstPtr &msg) {
    //std::cout << "====================  Image has get ========================" << endl;
    // =============== get cameraImage ======================== //
    cv::Mat cameraImage;
    cv_bridge::CvImagePtr input_bridge;
    try{
        input_bridge = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cameraImage = input_bridge->image;
    }
    catch (cv_bridge::Exception& ex){
        //std::cout << "Image prossing fail!!!!!!!!!!!!!!!!" << endl;
        return;
    }

    // ================ PointCloud to Image ==================== //
    std::vector<cv::Point2f> imageVecCC;
    std::vector<cv::Point3f> cloudVecCC;
    pcl::PointCloud<pcl::PointXYZ>::Ptr CC (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*Centroid, *CC, l_t_c);

    for (const auto& point : *CC) {
        if(point.z >=0)
            cloudVecCC.push_back(cv::Point3f(point.x, point.y, point.z));
    }

    cv::projectPoints(cloudVecCC, rVec, tVec, INTRINSIC_MAT, DIST_COEFFS, imageVecCC);

    for(size_t i=0; i<imageVecCC.size(); i++){
        cv::circle(cameraImage, imageVecCC[i], 12, CV_RGB(255,0,0), -1);
    }
    
    sensor_msgs::ImagePtr outputImage;
    outputImage = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cameraImage).toImageMsg();
    image_pub.publish(outputImage);
}

void time_callback(const std_msgs::Time::ConstPtr &msg) {
    bag_time = *msg;
    //std::cout << msg->data << std::endl;
}

void YOLO_boundingBox_callback(const darknet_ros_msgs::BoundingBoxes &msg) {
    if(isBusy) return;
    std::vector<YOLO_BoundingBox> tmp_YOLO_BB;
    for(int i = 0; i < msg.bounding_boxes.size(); i++)
    {
        YOLO_BoundingBox tmp_BB( msg.bounding_boxes[i].Class,
                                 msg.bounding_boxes[i].xmin,
                                 msg.bounding_boxes[i].xmax,
                                 msg.bounding_boxes[i].ymin,
                                 msg.bounding_boxes[i].ymax  );
        tmp_YOLO_BB.push_back(tmp_BB);
    }
    YOLO_BB = tmp_YOLO_BB;
}




int main(int argc, char** argv)
{
    // ROS init
    ros::init (argc,argv,"kf_tracker");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);

    //test();
    isBusy = false;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber cameraImage_sub = it.subscribe("/darknet_ros/detection_image", 10, cameraImage_callback);
    ros::Subscriber bagTime_sub                 = nh.subscribe("/bag_time", 10, time_callback);
    ros::Subscriber boundingBox_sub             = nh.subscribe("/darknet_ros/bounding_boxes",1, YOLO_boundingBox_callback);
    ros::Subscriber centroid_sub                = nh.subscribe("centroid", 10, centroid_callback);

    id_marker_array_pub = nh.advertise <visualization_msgs::MarkerArray> ("/id_marker_array", 10);
    Complete_pub        = nh.advertise <std_msgs::Int32MultiArray> ("/complete_check", 1);
    image_pub           = nh.advertise <sensor_msgs::Image> ("visualization_msgs/Image",1);


    initializeGlobalParams();

    while(ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

}
