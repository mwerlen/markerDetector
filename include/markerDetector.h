#ifndef MARKERDETECTOR_H_
#define MARKERDETECTOR_H_

#include <vector>
#include <memory>

#include <opencv/cv.h>

#include "MarkerDetectorConfig.h"

namespace markerDetector {

    typedef std::vector<cv::Point2i> Contour;
    
    typedef cv::RotatedRect Ellipse;
    
    struct EllipsesCluster {
        cv::Point2f center;
        int rep; // the representative circle
        std::vector<int> ellipsesIds;
    };

    /**
     * A target
     * includes temporary and final results of the detect and measure process
     */

    class Target {
      public:
        // detect stage
        bool detected; /**< if the target has been detected in the image **/
        Ellipse inner, outer; /**< inner and outer circle circle approximations **/

        // measure stage
        bool measured; /**< if the accurate measurement step succeded **/

        // temporary, should become a rotation and a translation
        double cx, cy;

        MarkerModel* markerModel;
        Target();
    };

    class MarkerDetector {
      protected:
        MarkerDetectorConfig _cfg;

      public:
        std::vector<Target> detectAndMeasure(const cv::Mat &image, cv::Mat &debug) ;
        void detectEdges(const cv::Mat& raw, cv::Mat& edges);
        void detectContours(const cv::Mat &edges, std::vector<std::vector<cv::Point>> &ctrs);
        void filterEllipses(const std::vector<Ellipse> &ellipses, std::vector<Ellipse> &filteredEllipses);
        std::vector<EllipsesCluster> clusterEllipses(const std::vector<Ellipse> &ellipses, cv::Mat &debug);

        MarkerDetector(const MarkerDetectorConfig &cfg) :
            _cfg(cfg) {
        }
    };

}

#endif /* MARKERDETECTOR_H_ */
