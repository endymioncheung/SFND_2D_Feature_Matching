/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
// Camera-based processing loop such as
// feature detection, descriptor, extraction, matching 
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // Data location
    string dataPath = "../";

    // Camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // First file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // No. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    bool bSaveImg = false;        // save the keypoints detection and matching results

    /* MAIN LOOP OVER ALL IMAGES */

    vector<int> num_kPts, num_matches;
    vector<float> t_detKeypoints, t_matches;
    for (size_t imgIndex = 0; imgEndIndex <= 1; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // Assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        DataFrame frame;
        frame.cameraImg = imgGray;
        
        if (dataBuffer.size() <  dataBufferSize)
        {
            // Push image into data frame buffer
            // when ring buffer is not full yet
            dataBuffer.push_back(frame);
        } else {
            // Remove the oldest data frame
            // when ring buffer is full
            dataBuffer.erase(dataBuffer.begin());
            /// Insert image into data frame buffer
            dataBuffer.push_back(frame);
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // Extract 2D keypoints from current image
        // Create empty feature list for current image
        // which holds all of the keypoints detected in the image
        vector<cv::KeyPoint> keypoints;
        string detectorType = "SHITOMASI";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double t_detKeypoint = (double)cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        t_detKeypoint = ((double)cv::getTickCount() - t_detKeypoint) / cv::getTickFrequency();
        num_kPts.push_back(keypoints.size());
        t_detKeypoints.push_back(t_detKeypoint);
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // Only keep keypoints on the preceding vehicle
        // (i.e. inside the bounding rectangle)
        bool bFocusOnVehicle = true;
        // Rectangle which always encloses the directly preceding vehicle
        cv::Rect vehicleRect(535, 180, 180, 150); // arguments: [cx,cy,w,h]
        if (bFocusOnVehicle)
        {
            for (auto it = keypoints.begin(); it < keypoints.end(); it++)
            {
                // Method 1: Remove keypoints that are outside the region of interest
                // cv::KeyPoint curr_keypoint;
                // curr_keypoint.pt = cv::Point2f((*it).pt.x, (*it).pt.y);
                // int pt_x     = curr_keypoint.pt.x;
                // int pt_y     = curr_keypoint.pt.y;
                // int left_x   = vehicleRect.x;
                // int right_x  = vehicleRect.x + vehicleRect.width;
                // int top_y    = vehicleRect.y + vehicleRect.height;
                // int bottom_y = vehicleRect.y;
                // if (left_x > pt_x || pt_x > right_x || pt_y > top_y || pt_y < bottom_y)
                // {
                //     keypoints.erase(it);
                // }

                // Method 2: Remove keypoints that are outside the region of interest
                if (!vehicleRect.contains(it->pt))
                {
                    keypoints.erase(it);
                }
            }
            // std::cout << "Focus on vehicle keypoints = " << keypoints.size() << "\n";
        }

        //// EOF STUDENT ASSIGNMENT

        // Optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // Push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorAlgoType = "BRISK"; // BRISK, ORB, AKAZE, SIFT, BRIEF, FREAK
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorAlgoType);
        //// EOF STUDENT ASSIGNMENT

        // Push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        // Wait until at least two images have been processed
        // only entered once the data buffer size exceeds a single element
        // because we want to match keypoints of TWO images, one image won't work
        // for keypoint matching
        if (dataBuffer.size() > 1)
        {
            /* MATCH KEYPOINT DESCRIPTORS */
            // Create vector to store the descriptors match results
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY (i.e. for BRISK, ORB), DES_HOG (i.e. for SITF)
            string selectorType = "SEL_NN";       // SEL_NN (select nearest neighbor), SEL_KNN (select nearest kth neighbor) 

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            double t_match = (double)cv::getTickCount();
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);
            t_match = ((double)cv::getTickCount() - t_match) / cv::getTickFrequency();
            num_matches.push_back(matches.size());
            t_matches.push_back(t_match);
            //// EOF STUDENT ASSIGNMENT

            // Store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // Visualize matches between current and previous image
            bVis = false;
            
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            if (bVis)
            {
                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;

            bSaveImg = false;
            if (bSaveImg)
            {
                string savedImgFullFilename = "../images/results/" + imgNumber.str() + "_" + detectorType + "_" + descriptorAlgoType + "_" + matcherType + "_" + descriptorType + "_" + selectorType + imgFileType;
                cout << "Saving results to ... " << savedImgFullFilename << endl << endl;
                cv::imwrite(savedImgFullFilename, matchImg);
            }
        }

    } // EOF loop over all images

    float avgkPts = 0.0f;
    float sumkPts = 0.0f;
    for (int i=0; i< num_kPts.size(); i++)
    {
        sumkPts += num_kPts[i];
    }
    avgkPts = sumkPts / num_kPts.size();
    cout << "Average detected keypoints = " << int(avgkPts) << endl;

    float avgMatches = 0.0f;
    float sumMatches = 0.0f;
    for (int i=0; i< num_matches.size(); i++)
    {
        sumMatches += num_matches[i];
    }
    avgMatches = sumMatches / num_matches.size();
    cout << "Average matched keypoints = " << int(avgMatches) << endl;

    float avg_t_detKeypoints = 0.0f;
    float sum_t_detKeypoints = 0.0f;
    for (int i=0; i< t_detKeypoints.size(); i++)
    {
        // cout << t_detKeypoints[i] << endl;
        sum_t_detKeypoints += t_detKeypoints[i];
    }
    avg_t_detKeypoints = sum_t_detKeypoints / t_detKeypoints.size();
    cout << "Average time to detect keypoints = " << (avg_t_detKeypoints * 1000 / 1.0) << " ms" << endl;

    float avg_t_matches = 0.0f;
    float sum_t_matches = 0.0f;
    for (int i=0; i< t_matches.size(); i++)
    {
        // cout << t_matches[i] << endl;
        sum_t_matches += t_matches[i];
    }
    avg_t_matches = sum_t_matches / t_matches.size();
    cout << "Average time to extract keypoints = " << (avg_t_matches * 1000 / 1.0) << " ms" << endl;
    // cout << avgkPts << " " << avgMatches << " " << avg_t_detKeypoints*1000 << " " << avg_t_matches*1000 << endl;
    return 0;
}