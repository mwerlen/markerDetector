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
        Ellipse outer;
        Ellipse inner;
        float diffW;
        float diffH;
        float distance;
    };


    /**
     * A target
     * includes temporary and final results of the detect and measure process
     */

    class Target {
      public:
        
        // inner and outer target ellipse approximations
        Ellipse inner, outer;
        
        // Center in image
        double cx, cy;
        
        // MarkerModel detected
        int markerModelId;

        //Correlation score
        float correlationScore;

        Target(){};
    };

    class MarkerDetector {
      protected:
        MarkerDetectorConfig _cfg;

      public:
        void detectAndMeasure(const cv::Mat &image, std::vector<Target> &targets, cv::Mat &debug) ;
        void detectEdges(const cv::Mat& raw, cv::Mat& edges);
        void detectContours(const cv::Mat &edges, std::vector<std::vector<cv::Point>> &ctrs);
        void contoursToEllipses(const std::vector<Contour> &contours, std::vector<Ellipse> &ellipses);
        void filterClusters(const std::vector<EllipsesCluster> &clusters, std::vector<EllipsesCluster> &filteredClusters);
        std::vector<EllipsesCluster> clusterEllipses(const std::vector<Ellipse> &ellipses);

        //Debug
        void debugCluster(const EllipsesCluster &cluster, cv::Mat &debug);
        void debugContours(const std::vector<Contour> &contours, cv::Mat &debug);
        void debugTargets(const std::vector<Target> &target, cv::Mat &debug);
        void debugSignalContour(const Contour &contour, cv::Mat &debug);

        MarkerDetector(const MarkerDetectorConfig &cfg) :
            _cfg(cfg) {
        }
    };

}

#endif /* MARKERDETECTOR_H_ */
