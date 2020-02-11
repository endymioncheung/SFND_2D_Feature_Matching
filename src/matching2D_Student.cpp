#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // Configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // Perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { 
        // Nearest neighbor (best match)
        // Find the best match for each descriptor in desc
        matcher->match(descSource, descRef, matches); 
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // Select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // Detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {
        //...
    }

    // Perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Compute detector parameters based on image size
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Visualize results
    // string windowName = "Harris Corner Detector Response Matrix";
    // cv::namedWindow(windowName, 4);
    // cv::imshow(windowName, dst_norm_scaled);
    // cv::waitKey(0);

    // Locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // Max. permissible overlap between two features in [%], used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            // Harris corner detector response for the pixel
            int response = (int)dst_norm.at<float>(j, i);
            // Only store points above a threshold
            if (response > minResponse)
            { 
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // Perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        // if overlap is > maxOverlap AND response is higher for new kpt
                        if (newKeyPoint.response > (*it).response)
                        {
                            // Replace old key point with new one
                            // quit loop over keypoints
                            *it = newKeyPoint; 
                            break;             
                        }
                    }
                }

                // Only add new key point if no overlap has been found in previous NMS
                if (!bOverlap)
                {
                    // Store new keypoint in dynamic list
                    keypoints.push_back(newKeyPoint); 
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Compute detector parameters based on image size
    int blockSize = 4;       // Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // Max permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // Max num of keypoints

    double qualityLevel = 0.01; // Minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // Add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // Call modern keypoint detetection algorithms
    if (detectorType.compare("SHITOMASI") == 0)
        detKeypointsShiTomasi(keypoints,img,bVis);
    if (detectorType.compare("HARRIS") == 0)
        detKeypointsHarris(keypoints,img,bVis);
    else if (detectorType.compare("FAST") == 0)
        detKeypointsFAST(keypoints,img,bVis);
    else if (detectorType.compare("BRISK") == 0)
        detKeypointsBRISK(keypoints,img,bVis);
    else if (detectorType.compare("ORB") == 0)
        detKeypointsORB(keypoints,img,bVis);
    else if (detectorType.compare("AKAZE") == 0)
        detKeypointsAKAZE(keypoints,img,bVis);
    else if (detectorType.compare("SIFT") == 0)
        detKeypointsSIFT(keypoints,img,bVis);
    else
        // Default detector to use the classical Shi-Tomasi
        // detector if none of the specified detector
        // type is matched
        detKeypointsShiTomasi(keypoints,img,bVis);
    return;
}

// Detect keypoints in image using the modern FAST detector
void detKeypointsFAST(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // OpenCV docs: https://docs.opencv.org/4.1.1/df/d74/classcv_1_1FastFeatureDetector.html

    // Difference between intensity of the central pixel and pixels of a circle around this pixel
    int threshold = 30;
    // Perform non-maxima suppression on keypoints
    bool bNMS = true;
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the modern BRISK detector
void detKeypointsBRISK(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // OpenCV docs: https://docs.opencv.org/4.1.1/de/dbf/classcv_1_1BRISK.html

    int threshold = 30;
    int octaves = 3;
    float patternScale = 1.0f;
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create(threshold, octaves, patternScale);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the modern ORB detector
void detKeypointsORB(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // OpenCV docs: https://docs.opencv.org/4.1.1/db/d95/classcv_1_1ORB.html
    //  More about cv::ORB::ScoreType scoreType:
    //      - HARRIS_SCORE = 0
    //      The default HARRIS_SCORE means that Harris algorithm is used to rank features 
    //      (the score is written to KeyPoint::score and is used to retain best nfeatures features); 
    //      - FAST_SCORE = 1
    //      FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, 
    //      but it is a little faster to compute.

    int 	nfeatures = 500;
    float 	scaleFactor = 1.2f;
    int 	nlevels = 8;
    int 	edgeThreshold = 31;
    int 	firstLevel = 0;
    int 	WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int 	patchSize = 31;
    int 	fastThreshold = 20;
    
    cv::Ptr<cv::ORB> detector = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the modern AKAZE detector
void detKeypointsAKAZE(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // OpenCV docs: https://docs.opencv.org/4.1.1/d8/d30/classcv_1_1AKAZE.html
        
    // More about cv::AKAZE::DescriptorType descriptor_type:
    //      - DESCRIPTOR_KAZE = 3
    //      - DESCRIPTOR_KAZE_UPRIGHT = 2 (Upright descriptors, not invariant to rotation)
    //      - DESCRIPTOR_MLDB = 5
    //      - DESCRIPTOR_MLDB_UPRIGHT = 4 (Upright descriptors, not invariant to rotation)

    // More about cv::KAZE::DiffusivityType diffusivity: 
    //      - DIFF_PM_G1
    //      - DIFF_PM_G2
    //      - DIFF_WEICKERT
    //      - DIFF_CHARBONNIER

    cv::AKAZE::DescriptorType   descriptor_type = cv::AKAZE::DESCRIPTOR_KAZE;
    int 	descriptor_size = 0;
    int 	descriptor_channels = 3;
    float 	threshold = 0.001f;
    int 	nOctaves = 4;
    int 	nOctaveLayers = 4;
    cv::KAZE::DiffusivityType   diffusivity = cv::KAZE::DIFF_PM_G2; 
    
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the modern SIFT detector
void detKeypointsSIFT(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // OpenCV docs: https://docs.opencv.org/4.1.1/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
    
    int 	nfeatures = 0;
    int 	nOctaveLayers = 3;
    double 	contrastThreshold = 0.04;
    double 	edgeThreshold = 10;
    double 	sigma = 1.6;
    
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}