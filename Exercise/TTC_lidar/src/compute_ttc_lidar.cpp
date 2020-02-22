#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>


#include "dataStructures.h"
#include "structIO.hpp"

using namespace std;

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double &TTC)
{
    // Auxiliary variables
    const double dT = 0.1;        // Time between two measurements [s]
    const double laneWidth = 4.0; // Assumed road lane width [m]

    // Find closest distance to 3D Lidar points within ego lane
    // in the previous and current frame
    double minXPrev = 1e9, minXCurr = 1e9;;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
        if (abs(it->y) <= laneWidth / 2.0)
            minXPrev = min(minXPrev,it->x);
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
        if (abs(it->y) <= laneWidth / 2.0)
            minXCurr = min(minXCurr,it->x);

    // Compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

int main()
{
    // Read Lidar points in the previous and current frame
    std::vector<LidarPoint> currLidarPts, prevLidarPts;
    readLidarPts("../dat/C22A5_currLidarPts.dat", currLidarPts);
    readLidarPts("../dat/C22A5_prevLidarPts.dat", prevLidarPts);

    // Compute TTC (Time To Collision) based on Lidar points
    double ttc;
    computeTTCLidar(prevLidarPts, currLidarPts, ttc);
    cout << "ttc = " << ttc << "s" << endl;
}