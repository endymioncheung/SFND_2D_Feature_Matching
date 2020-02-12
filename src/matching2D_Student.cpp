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
        cout << "Brute-force matching";
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { 
            // OpenCV bug workaround : convert binary descriptors to floating point 
            // due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        // FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching";
    } else {
        // Default to brute force if no matching matcher option is found
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "Brute-force matching";
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
        // for each key frame get the two best matches in the second key frame
        double t = (double)cv::getTickCount();
        // k-nearest-neighbor matching with the two best matches
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // Filter matches using descriptor distance ratio test.
        // Descriptor distance ratio test to compare two best matches
        // select those who is greater than the threshold
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            // Match found if the descriptor distance ratio of
            // the source image and reference image is within the threshold
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (kNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // Select appropriate descriptor
    // Template pointer to a generic descriptor extractors structure
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        /* 
        BRISK: Binary robust invariant scalable keypoints
        OpenCV docs: https://docs.opencv.org/4.1.1/de/dbf/classcv_1_1BRISK.html

        Stefan Leutenegger, Margarita Chli, and Roland Yves Siegwart. Brisk: Binary robust invariant scalable keypoints. 
        In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2548–2555. IEEE, 2011.
        */

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // Detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) {
        // OpenCV docs: https://docs.opencv.org/3.4/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html

        int bytes = 32;                 // Length of the descriptor in bytes, valid values are: 16, 32 (default) or 64
        bool use_orientation = false;   // Sample patterns using keypoints orientation, disabled by default.
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes,use_orientation);
    } else if (descriptorType.compare("ORB") == 0) {
        /* 
        ORB: Oriented BRIEF
        OpenCV docs: https://docs.opencv.org/4.1.1/db/d95/classcv_1_1ORB.html

        The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response, 
        finds their orientation using first-order moments and computes the descriptors using BRIEF (where the coordinates 
        of random point pairs (or k-tuples) are rotated according to the measured orientation).

        Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: an efficient alternative to sift or surf. 
        In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2564–2571. IEEE, 2011.
        */

        int 	nfeatures = 500;    // The maximum number of features to retain
        float 	scaleFactor = 1.2f; // Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, 
                                    // where each next level has 4x less pixels than the previous but such a big scale factor 
                                    // will degrade feature matching scores dramatically. On the other hand, too close to 1 
                                    // scale factor will mean that to cover certain scale range you will need more pyramid 
                                    // levels and so the speed will suffer.
        int 	nlevels = 8;        // The number of pyramid levels. The smallest level will have linear size equal to 
                                    // input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        int 	edgeThreshold = 31; // The size of the border where the features are not detected. It should roughly match the patchSize parameter
        int 	firstLevel = 0;     // The level of pyramid to put source image to. Previous layers are filled with upscaled source image
        int 	WTA_K = 2;          // The number of points that produce each element of the oriented BRIEF descriptor. 
                                    // The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, 
                                    // so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points 
                                    // (of course, those point coordinates are random, but they are generated from the pre-defined seed, 
                                    // so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), 
                                    // find point of maximum brightness and output index of the winner (0, 1 or 2). Such output 
                                    // will occupy 2 bits, and therefore it will need a special variant of Hamming distance, 
                                    // denoted as NORM_HAMMING2 (2 bits per bin). 
                                    // When WTA_K=4, we take 4 random points to compute each bin 
                                    // (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // The default HARRIS_SCORE means that Harris algorithm is used to rank features 
                                                              // (the score is written to KeyPoint::score and is used to retain best nfeatures features); 
                                                              // FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints 
                                                              // but it is a little faster to compute.
        int 	patchSize = 31;     // The size of the patch used by the oriented BRIEF descriptor. 
                                    // Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        int 	fastThreshold = 20; // The fast threshold
    
        extractor = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
    } else if (descriptorType.compare("FREAK") == 0) {
        /* 
        FREAK: Fast Retina Keypoint
        OpenCV docs: https://docs.opencv.org/4.1.1/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html

        The algorithm propose a novel keypoint descriptor inspired by the human visual system and more precisely the retina, coined Fast Retina Key- point (FREAK). 
        A cascade of binary strings is computed by efficiently comparing image intensities over a retinal sampling pattern. 
        FREAKs are in general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK. 
        They are competitive alternatives to existing keypoints in particular for embedded applications.

        Alexandre Alahi, Raphael Ortiz, and Pierre Vandergheynst. Freak: Fast retina keypoint. 
        In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on, pages 510–517. Ieee, 2012.
        */

        bool 	orientationNormalized = true;   // Enable orientation normalization
        bool 	scaleNormalized = true;         // Enable scale normalization
        float 	patternScale = 22.0f;           // Scaling of the description pattern.
        int 	nOctaves = 4;                   // Number of octaves covered by the detected keypoints
        
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized,scaleNormalized,patternScale,nOctaves);
    } else if (descriptorType.compare("AKAZE") == 0) {
        /* 
        AKAZE descriptors can only be used with `KAZE` or `AKAZE` keypoints. This class is thread-safe.
        OpenCV docs: https://docs.opencv.org/4.1.1/d8/d30/classcv_1_1AKAZE.html

        Note:
        When you need descriptors use `Feature2D::detectAndCompute`, which provides better performance. 
        When using `Feature2D::detect` followed by `Feature2D::compute` scale space pyramid is computed twice.
        `AKAZE` implements T-API. When image is passed as `UMat` some parts of the algorithm will use OpenCL.
        
        [ANB13] Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien Bartoli. 
        In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
        */
    
        cv::AKAZE::DescriptorType   descriptor_type = cv::AKAZE::DESCRIPTOR_KAZE;   // Type of the extracted descriptor
                                                                                    // cv::AKAZE::DescriptorType descriptor_type:
                                                                                    // - DESCRIPTOR_KAZE = 3
                                                                                    // - DESCRIPTOR_KAZE_UPRIGHT = 2 (Upright descriptors, not invariant to rotation)
                                                                                    // - DESCRIPTOR_MLDB = 5
                                                                                    // - DESCRIPTOR_MLDB_UPRIGHT = 4 (Upright descriptors, not invariant to rotation)
        int 	descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
        int 	descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3)
        float 	threshold = 0.001f;         // Detector response threshold to accept point
        int 	nOctaves = 4;               // Maximum octave evolution of the image
        int 	nOctaveLayers = 4;          // Default number of sublevels per scale level
        cv::KAZE::DiffusivityType   diffusivity = cv::KAZE::DIFF_PM_G2;             // Diffusivity type
                                                                                    // cv::KAZE::DiffusivityType diffusivity: 
                                                                                    // - DIFF_PM_G1
                                                                                    // - DIFF_PM_G2
                                                                                    // - DIFF_WEICKERT
                                                                                    // - DIFF_CHARBONNIER

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);

    } else if (descriptorType.compare("SIFT") == 0) {
        /*
        SIFT: Scale Invariant Feature Transform
        OpenCV docs: https://docs.opencv.org/4.1.1/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

        David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.
        */

        int 	nfeatures = 0;              // The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        int 	nOctaveLayers = 3;          // The number of layers in each octave. 3 is the value used in D. Lowe paper. 
                                            // The number of octaves is computed automatically from the image resolution
        double 	contrastThreshold = 0.04;   // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
                                            // The larger the threshold, the less features are produced by the detector
        double 	edgeThreshold = 10;         // The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, 
                                            // i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained)
        double 	sigma = 1.6;                // The sigma of the Gaussian applied to the input image at the octave #0. 
                                            // If your image is captured with a weak camera with soft lenses, you might want to reduce the number
        
        extractor = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);

    } else {
        // Default to use BRISK if no matching descriptor selector specified
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // Detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
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
    int blockSize = 2;     // For every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // Aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // Minimum value for a corner in the 8bit scaled response matrix
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
    /*
    FAST: Fast feature detector
    OpenCV docs: https://docs.opencv.org/4.1.1/df/d74/classcv_1_1FastFeatureDetector.html
    */
    
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
    /* 
    BRISK: Binary robust invariant scalable keypoints
    OpenCV docs: https://docs.opencv.org/4.1.1/de/dbf/classcv_1_1BRISK.html

    Stefan Leutenegger, Margarita Chli, and Roland Yves Siegwart. Brisk: Binary robust invariant scalable keypoints. 
    In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2548–2555. IEEE, 2011.
    */

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
    /* 
    ORB: Oriented BRIEF
    OpenCV docs: https://docs.opencv.org/4.1.1/db/d95/classcv_1_1ORB.html

    The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response, 
    finds their orientation using first-order moments and computes the descriptors using BRIEF (where the coordinates 
    of random point pairs (or k-tuples) are rotated according to the measured orientation).

    Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: an efficient alternative to sift or surf. 
    In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2564–2571. IEEE, 2011.
    */

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
    /* 
    AKAZE descriptors can only be used with `KAZE` or `AKAZE` keypoints. This class is thread-safe.
    OpenCV docs: https://docs.opencv.org/4.1.1/d8/d30/classcv_1_1AKAZE.html

    Note:
    When you need descriptors use `Feature2D::detectAndCompute`, which provides better performance. 
    When using `Feature2D::detect` followed by `Feature2D::compute` scale space pyramid is computed twice.
    `AKAZE` implements T-API. When image is passed as `UMat` some parts of the algorithm will use OpenCL.
    
    [ANB13] Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien Bartoli. 
    In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
    */
        
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
    /*
    SIFT: Scale Invariant Feature Transform
    OpenCV docs: https://docs.opencv.org/4.1.1/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

    David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.
    */
    
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