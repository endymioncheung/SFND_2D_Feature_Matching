#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


using namespace std;

void changePixels()
{
    // create matrix
    int nrows = 480, ncols = 640;
    cv::Mat m1_8u;
    m1_8u.create(nrows, ncols, CV_8UC1); // single-channel matrix with 8bit unsigned elements
    m1_8u.setTo(cv::Scalar(0));          //black

    // cv::Mat m3_8u;
    // m3_8u.create(nrows, ncols, CV_8UC3); // three-channel matrix with 8bit unsigned elements
    // m3_8u.setTo(cv::Scalar(0, 0, 0));  // blue

    for (int r = 230; r < 250; r++)
    {
        // STUDENT TASK : loop over all columns and set matrix elements to 255
        for (int col = 0; col < m1_8u.cols ; col++)
            m1_8u.at<unsigned char>(r,col) = 255;
        // for (int col = 0; col < m3_8u.cols ; col++)
        //     m3_8u.at<cv::Scalar>(r,col) = cv::Scalar(255, 255, 255);
    }
    
    // show result
    string windowName = "First steps in OpenCV";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, m1_8u);
    cv::waitKey(0); // wait for keyboard input before continuing
}


int main()
{
    changePixels();
    return 0;
}