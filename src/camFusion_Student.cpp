
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<float> euclideanDistances;
    std::vector<cv::DMatch> roiMatches;

    for (const auto match : kptMatches)
    {
        cv::KeyPoint keyPointPrev = kptsPrev.at(match.queryIdx);
        cv::KeyPoint keyPointCurr = kptsCurr.at(match.trainIdx);

        //check if keypoints are inside the roi, calculate eucliden distance and store probable matches
        if (boundingBox.roi.contains(keyPointCurr.pt) && boundingBox.roi.contains(keyPointPrev.pt))
        {
            roiMatches.push_back(match);
            cv::Point2f diff = keyPointCurr.pt - keyPointPrev.pt;
            float euclideanDistance = cv::norm(diff);
            euclideanDistances.push_back(euclideanDistance);
        }
    }

    //std::cout << "ROI Matching bounding boxes = " << roiMatches.size() << "\n";
    std::sort(euclideanDistances.begin(), euclideanDistances.end());

    //Using IQR to remove outliers.
    int indexQ1 = floor(euclideanDistances.size() / 4);
    int indexQ3 = floor((euclideanDistances.size() * 3) / 4);

    float Q1 = euclideanDistances.at(indexQ1);
    float Q3 = euclideanDistances.at(indexQ3);
    float IQR = Q3 - Q1;

    for (auto match : roiMatches)
    {
        cv::KeyPoint keyPointPrev = kptsPrev.at(match.queryIdx);
        cv::KeyPoint keyPointCurr = kptsCurr.at(match.trainIdx);
        cv::Point2f diff = keyPointCurr.pt - keyPointPrev.pt;
        float euclideanDistance = cv::norm(diff);

        if ((Q1 - 1.5 * IQR) <= euclideanDistance && euclideanDistance <= (Q3 + 1.5 * IQR))
        {
            boundingBox.kptMatches.push_back(match);
        }
    }

    //std::cout << "Final Matching boxes roi = " << boundingBox.kptMatches.size() << "\n";
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distanceRatios;
    for (int i = 0; i < kptMatches.size() - 1; i++) //outer loop
    {
        cv::KeyPoint keypointOuterCurr = kptsCurr.at(kptMatches.at(i).trainIdx);
        cv::KeyPoint keypointOuterPrev = kptsPrev.at(kptMatches.at(i).queryIdx);

        for (int j = 1; j < kptMatches.size(); j++) //inner loop
        {
            double minDist = 100.0;

            cv::KeyPoint keypointInnerCurr = kptsCurr.at(kptMatches.at(j).trainIdx);
            cv::KeyPoint keypointInnerPrev = kptsPrev.at(kptMatches.at(j).queryIdx);

            double distanceCurr = cv::norm(keypointInnerCurr.pt - keypointOuterCurr.pt);
            double distancePrev = cv::norm(keypointInnerPrev.pt - keypointOuterPrev.pt);

            if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= minDist)
            {
                double distanceRatio = distanceCurr / distancePrev;
                distanceRatios.push_back(distanceRatio);
            }
        }
    }
    if (distanceRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    //sort distance Ratios to find median distRatio
    std::sort(distanceRatios.begin(), distanceRatios.end());
    long medianRatioIndex = floor(distanceRatios.size() / 2.0);
    double medianDistanceRatio = distanceRatios.size() % 2 == 0 ? (distanceRatios.at(medianRatioIndex - 1) + distanceRatios.at(medianRatioIndex)) / 2.0 : distanceRatios.at(medianRatioIndex);

    double dT = 1 / frameRate; //time between consecutive frames
    TTC = -dT / (1 - medianDistanceRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate; //time between two consecutive measurements

    double egoLaneWidth{5.0}; //consider lidar points only in the ego lane to avoid outliers

    //Use IQR to remove any possible outlier
    std::vector<float> distributionXMinPrev;
    std::vector<float> distributionXMinCurr;

    float xMinPrev{1e8};
    float xMinCurr{1e8};

    for (const auto &lidarPoint : lidarPointsPrev)
    {
        float y = lidarPoint.y;
        float x = lidarPoint.x;

        if (std::abs(y) <= egoLaneWidth / 2.0)
        {
            distributionXMinPrev.push_back(x);
        }
    }

    std::sort(distributionXMinPrev.begin(), distributionXMinPrev.end());
    float Q1 = distributionXMinPrev.at(distributionXMinPrev.size() / 4);
    float Q3 = distributionXMinPrev.at((3 * distributionXMinPrev.size()) / 4);
    float IQR = Q3 - Q1;
    for (float xprev : distributionXMinPrev)
    {
        if (xprev > (Q1 - 1.5 * IQR))
        {
            xMinPrev = xprev;
            break;
        }
    }

    xMinPrev = (Q3 + Q1) / 2.0;
    for (const auto &lidarPoint : lidarPointsCurr)
    {
        float y = lidarPoint.y;
        float x = lidarPoint.x;

        if (std::abs(y) <= egoLaneWidth / 2.0)
        {
            distributionXMinCurr.push_back(x);
        }
    }

    std::sort(distributionXMinCurr.begin(), distributionXMinCurr.end());
    Q1 = distributionXMinCurr.at(distributionXMinCurr.size() / 4);
    Q3 = distributionXMinCurr.at((3 * distributionXMinCurr.size()) / 4);
    IQR = Q3 - Q1;
    for (float xcurr : distributionXMinCurr)
    {
        if (xcurr > (Q1 - 1.5 * IQR))
        {
            xMinCurr = xcurr;
            break;
        }
    }
    xMinCurr = (Q3 + Q1) / 2.0;
    cout << "XminPRev : " << xMinPrev << " | XminCurr : " << xMinCurr << "\n";
    TTC = (xMinCurr * dT) / (xMinPrev - xMinCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    /*
        Check if the match is present in the roi of both currFrame and prevFrame. For such matches, populate a map
        that has the id of prevBoundingBox as key. Value is a vector of counts, indexed by id of currBoundingBox.
        Then iterate over the vectors for every prevId and find the best currId.
    */
    //Find all possible pairs of <prev_frame_id, curr_frame_id>, store the count in a map. prev_frame_id -> {curr_frame_id, count}
    int prevBoundingBoxesSize = prevFrame.boundingBoxes.size();
    int currBoundingBoxesSize = currFrame.boundingBoxes.size();

    std::map<int, std::vector<int>> prevIdCurrIdCountMap;
    for (auto match_ptr = matches.begin(); match_ptr != matches.end(); match_ptr++)
    {
        const cv::KeyPoint prevKeyPoint = prevFrame.keypoints.at(match_ptr->queryIdx);
        const cv::KeyPoint currKeyPoint = currFrame.keypoints.at(match_ptr->trainIdx);

        for (const auto &prevBoundingBox : prevFrame.boundingBoxes)
        {
            for (const auto &currBoundingBox : currFrame.boundingBoxes)
            {
                if (prevBoundingBox.roi.contains(prevKeyPoint.pt) && currBoundingBox.roi.contains(currKeyPoint.pt))
                {
                    //prev_id doesn't exist in the map
                    if (prevIdCurrIdCountMap.find(prevBoundingBox.boxID) == prevIdCurrIdCountMap.end())
                    {
                        std::vector<int> value(currBoundingBoxesSize, 0);
                        prevIdCurrIdCountMap.insert(std::make_pair(prevBoundingBox.boxID, value));
                    }
                    prevIdCurrIdCountMap[prevBoundingBox.boxID].at(currBoundingBox.boxID)++;
                }
            }
        }
    }

    //for every prev_id find the most occuring curr_id based on the index
    for (auto &idVectorPair : prevIdCurrIdCountMap)
    {
        int maxValue = 0;
        int bestCurrId = -1;
        int prevId = idVectorPair.first;
        for (int currId = 0; currId < idVectorPair.second.size(); currId++)
        {
            if (idVectorPair.second.at(currId) > maxValue)
            {
                maxValue = idVectorPair.second.at(currId);
                bestCurrId = currId;
            }
        }
        if (bestCurrId > -1)
        {
            bbBestMatches.insert(std::make_pair(prevId, bestCurrId));
        }
    }
}
