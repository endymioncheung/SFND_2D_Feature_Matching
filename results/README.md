# Camera Based 2D Feature Tracking

## Data Buffer

### MP.1 Data Buffer Optimization

Implement ring data buffer

```cpp
// Implement ring data buffer
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
```

## Keypoints

### MP.2 Keypoint Detection

Implement a selection of alternative detectors:

- HARRIS
- FAST
- BRISK
- ORB
- AKAZE
- SIFT

```cpp
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
```

### MP.3 Keypoint Removal

Remove all keypoints outside of a bounding box around the preceding vehicle. Box parameters you should use are : cx = 535, cy = 180, w = 180, h = 150.

```cpp
// Only keep keypoints on the preceding vehicle
// (i.e. inside the bounding rectangle)
bool bFocusOnVehicle = true;
// Rectangle which always encloses the directly preceding vehicle
cv::Rect vehicleRect(535, 180, 180, 150); // arguments: [cx,cy,w,h]
if (bFocusOnVehicle)
{
    for (auto it = keypoints.begin(); it < keypoints.end(); it++)
    {
        // Remove keypoints that are outside the region of interest
        if (!vehicleRect.contains(it->pt))
        {
            keypoints.erase(it);
        }
    }
    // std::cout << "Focus on vehicle keypoints = " << keypoints.size() << "\n";
}
```

## Descriptors

### MP.4 Keypoint Descriptors

Implement a variety of keypoint descriptors and make them selectable using the string 'descriptorType':

- BRIEF
- ORB
- FREAK
- AKAZE
- SIFT

```cpp
cv::Mat descriptors;
string descriptorAlgoType = "BRISK"; // BRISK, ORB, AKAZE, SIFT, BRIEF, FREAK
descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorAlgoType);

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

        int 	nfeatures = 99999;    // The maximum number of features to retain
        float 	scaleFactor = 1.2f; // Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, 
                                    // where each next level has 4x less pixels than the previous but such a big scale factor 
                                    // will degrade feature matching scores dramatically. On the other hand, too close to 1 
                                    // scale factor will mean that to cover certain scale range you will need more pyramid 
                                    // levels and so the speed will suffer.
        int 	nlevels = 3;        // The number of pyramid levels. The smallest level will have linear size equal to 
                                    // input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        int 	edgeThreshold = 30; // The size of the border where the features are not detected. It should roughly match the patchSize parameter
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
        int 	fastThreshold = 30; // The fast threshold
    
        extractor = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
    } else if (descriptorType.compare("FREAK") == 0) {
        /* 
        FREAK: Fast Retina Keypoint
        OpenCV docs: https://docs.opencv.org/4.1.1/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html

        The algorithm propose a novel keypoint descriptor inspired by the human visual system and more precisely the retina, coined Fast Retina Key-point (FREAK). 
        A cascade of binary strings is computed by efficiently comparing image intensities over a retinal sampling pattern. 
        FREAKs are in general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK. 
        They are competitive alternatives to existing keypoints in particular for embedded applications.

        Alexandre Alahi, Raphael Ortiz, and Pierre Vandergheynst. Freak: Fast retina keypoint. 
        In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on, pages 510–517. Ieee, 2012.
        */

        bool 	orientationNormalized = true;   // Enable orientation normalization
        bool 	scaleNormalized = true;         // Enable scale normalization
        float 	patternScale = 22.0f;           // Scaling of the description pattern.
        int 	nOctaves = 3;                   // Number of octaves covered by the detected keypoints
        
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
        int 	nOctaves = 3;               // Maximum octave evolution of the image
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
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}
```

### MP.5 Descriptor Matching and MP.6 Descriptor Distance Ratio

- Add FLANN as an alternative to brute-force matching
- Add K-Nearest-Neighbor selection

```cpp
double t_match = (double)cv::getTickCount();
matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                    matches, descriptorType, matcherType, selectorType);
t_match = ((double)cv::getTickCount() - t_match) / cv::getTickFrequency();
num_matches.push_back(matches.size());
t_matches.push_back(t_match);

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
        cout << "Brute-force matching" << endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // OpenCV bug workaround : convert binary descriptors to floating point
        // due to a bug in current OpenCV implementatio
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
        // FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching" << endl;;
    } else {
        // Default to brute force if no matching matcher option is found
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "Brute-force matching" << endl;
    }

    // Perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { 
        // Nearest neighbor (best match)
        // Find the best match for each descriptor in descSource
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); 
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "(NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k=2)
        // For each key frame get the two best matches in the second key frame
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
        cout << "(kNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}
```

## Performance

### MP.7 Performance Evaluation 1

Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT 

| Detector | Detected Keypoints | Execution time (ms) |
|----------|--------------------|---------------------|
| FAST     | 1787               | 0.70                |
| ORB      | 7683               | 9.07                |
| AKAZE    | 1246               | 51.23               |
| BRISK    | 2712               | 167.52              |
| STIFT    | 1386               | 126.89              |
| HARRIS   | 174                | 13.71               |

Distribution of their neighborhood size

- ORB: large uniform keypoints on strong vehicle features
- BRISK: non-uniform keypoints; strong in detecting vehicles regardless close or far
- FAST: small uniform keypoints; sparse detection on weak vehicle features
- SIFT: large highly non-uniform keypoints
- AKAZE: small uniform keypoints; weak in detecting nearby vehicle and more sensitive in detecting objects that are far away (i.e. too many non-vehicle objects)
- HARRIS: very small uniform keypoints; similar to BRISK but the keypoints are more sparse

### MP.8 Performance Evaluation 2

Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

| Detector | Matched Keypoints | % of Matched Keypoints | Execution time per matched Keypoint (us) |
|----------|-------------------|------------------------|------------------------------------------|
| ORB      | 1318              | 51.31%                 | 6.32                                     |
| BRIEF    | 1229              | 51.14%                 | 6.88                                     |
| BRISK    | 1154              | 51.10%                 | 7.68                                     |
| FREAK    | 890               | 46.26%                 | 7.39                                     |
| AKAZE    | 674               | 54.05%                 | 12.47                                    |
| STIFT    | 1257              | 53.2%                  | 20.53                                    |

### MP.9 Performance Evaluation 3

#### TOP 3 detectors

| Detector | Detected Keypoints | Execution time (ms) |
|----------|--------------------|---------------------|
| 1. FAST  | 1787               | 0.70                |
| 2. ORB   | 7683               | 9.07                |
| 3. AKAZE | 1246               | 51.23               |

#### TOP 3 descriptors

| Detector | Matched Keypoints | % of Matched Keypoints | Execution time per matched Keypoint (us) |
|----------|-------------------|------------------------|------------------------------------------|
| 1. ORB   | 1318              | 51.31%                 | 6.32                                     |
| 2. BRIEF | 1229              | 51.14%                 | 6.88                                     |
| 3. BRISK | 1154              | 51.10%                 | 7.68                                     |

#### Discusison on the detectors and descriptors

Above tables are my recommended top 3 choices for the keypoint detectors and keypoint decriptors.

With the selection of keypoints detectors, my selection criteria is number of detected keypoints first then the execution time to detect those keypoints. Therefore, I  highly recommend `cv::FAST` detector as it is the fastest keypoint out of tested detectors out of BRIEF, ORB, FREAK, AKAZE and SIFT while it still has above average performance (~1500 detections) in detecting keypoints . My second preferred detector would be `cv::ORB` because it is has the highest number of keypoints detection while being the second fastest to compute the keypoints detection. The third preferred detector would be AKAZE rather than BRISK despite BRISK being has 2x the keypoint detections than AKAZE, however BRISK takes 3x time longer to execute. The third preferred choice is really dependent on the criteria on the longest permissible time to execute and the quality of the keypoints detection as this could have significantly influence the selection of the detectors.

With the selection of the desciptors, almost all of the listed descriptors above have similar performance in matching keypoints, which is about around 50%. Therefore the focus in selecting a good descriptor then comes to the second selection criteria which is the executino time per matched keypoints. Therefore the recommended descriptor is ORB, followed by BRIEF and BRISK.