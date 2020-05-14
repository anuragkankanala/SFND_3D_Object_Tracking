
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }

    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }

    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }

    else if (descriptorType.compare("AKAZE") == 0)
    {
        int descriptorSize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, cv::KAZE::DIFF_PM_G2);
        // extractor = cv::AKAZE::create();
    }

    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
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

//Harris KeyPoint Detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    //detector params
    int blockSize{2};     //blocksize neighborhood
    int apertureSize{3};  //for sobel operator
    int minResponse{100}; //min value for a corner
    double k{0.04};       //Harris param

    //detect Harris corners
    cv::Mat dst;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    //normalize output
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    //Use non maximum supression to find the keypoints
    double maxOverlap = 0.0;
    for (size_t row = 0; row < dst_norm.rows; row++)
    {
        for (size_t col = 0; col < dst_norm.cols; col++)
        {
            int response = static_cast<int>(dst_norm.at<float>(row, col));
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(col, row); //row, col inverted here. columns align on x axis, rows on y axis
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                newKeyPoint.class_id = 0;

                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
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

//Modern Keypoint Detectors - FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    string windowName = "";

    //FAST detector
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "FAST detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        windowName = "FAST Corner Detector Results";
    }

    //BRISK detector
    else if (detectorType.compare("BRISK") == 0)
    {
        int thresh = 30;           //FAST/AGAST detection threshold score.
        int octaves = 3;           //detection octaves. Use 0 to do single scale.
        float patternScale = 1.0f; //apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(thresh, octaves, patternScale);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "BRISK detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        windowName = "BRISK Corner Detector Results";
    }

    else if (detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "ORB detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        windowName = "ORB Corner Detector Results";
    }

    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "AKAZE detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        windowName = "AKAZE Corner Detector Results";
    }

    else if (detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        windowName = "SIFT Corner Detector Results";
    }

    //To work with AKAZE algorithm, we need to set class_id = 0
    if (detectorType.compare("AKAZE") != 0)
    {
        // cout << "Setting class_id = 0\n";
        for (auto &keypoint : keypoints)
        {
            keypoint.class_id = 0;
        }
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}