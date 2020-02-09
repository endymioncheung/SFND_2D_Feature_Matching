#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;

void descKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // BRISK detector / descriptor
    // Other detectors can be also be used:
    // - BRIEF
	// - ORB
	// - FREAK
    // - KAZE
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    // Resulting output array (i.e. vector of KeyPoints)
    vector<cv::KeyPoint> kptsBRISK;

    double t = (double)cv::getTickCount();
    // Call BRISK function with grayscale image 
    // with the result data structure
    detector->detect(imgGray, kptsBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    // Local vicinity of a keypoint stored in a matrix
    cv::Mat descBRISK;
    t = (double)cv::getTickCount();
    // Execute the descriptor function which takes grayscale image
    // the keyPoints (can be from other detectors other than BRISK)
    descriptor->compute(imgGray, kptsBRISK, descBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results
    // Note: This is not the descriptors, it will be visualize in the next lesson
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsBRISK, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "BRISK Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);
    cv::waitKey(0);

    // TODO: Add the SIFT detector / descriptor, compute the 
    // time for both steps and compare both BRISK and SIFT
    // with regard to processing speed and the number and 
    // visual appearance of keypoints.
    // Visit OpenCV docs https://docs.opencv.org/4.1.1/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    // to learn to use SIFT
    detector = cv::xfeatures2d::SIFT::create();
    vector<cv::KeyPoint> kptsSIFT;
    
    t = (double)cv::getTickCount();
    detector->detect(imgGray, kptsSIFT);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n= " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    descriptor = cv::xfeatures2d::SIFT::create();
    // Local vicinity of a keypoint stored in a matrix
    cv::Mat descSIFT;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, kptsSIFT, descSIFT);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results
    visImage = img.clone();
    cv::drawKeypoints(img, kptsSIFT, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "SIFT Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

int main()
{
    descKeypoints1();
    return 0;
}