#ifndef STRUCTURES_H_
#define STRUCTURES_H_

#include <vector>

#include <opencv/cv.h>

namespace markerDetector {

    typedef std::vector<cv::Point2i> Contour;
    
    typedef cv::RotatedRect Ellipse;
    
    struct EllipsesCluster {
        cv::Point2f center;
        Ellipse outer;
        Ellipse inner;
        float diffW;
        float diffH;
        float distance;
    };
}
#endif /* STRUCTURES_H_ */
