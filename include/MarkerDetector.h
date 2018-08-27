#ifndef MARKERDETECTOR_H_
#define MARKERDETECTOR_H_

#include <vector>
#include <memory>

#include <opencv/cv.h>

#include "MarkerDetectorConfig.h"
#include "SignalReader.h"
#include "Structures.h"
#include "Target.h"

namespace markerDetector {

    class MarkerDetector {
      protected:
        MarkerDetectorConfig _cfg;
        SignalReader reader;

      public:
        void detectAndMeasure(const cv::Mat &image, std::vector<Target> &targets, cv::Mat &debug);
        void detectEdges(const cv::Mat& raw, cv::Mat& edges);
        void detectContours(const cv::Mat &edges, std::vector<std::vector<cv::Point>> &ctrs);
        void contoursToEllipses(const std::vector<Contour> &contours, std::vector<Ellipse> &ellipses);
        void filterClusters(const std::vector<EllipsesCluster> &clusters, std::vector<EllipsesCluster> &filteredClusters, const cv::Mat &image);
        std::vector<EllipsesCluster> clusterEllipses(const std::vector<Ellipse> &ellipses);

        //Debug
        void debugCluster(const EllipsesCluster &cluster, cv::Mat &debug);
        void debugContours(const std::vector<Contour> &contours, cv::Mat &debug);
        void debugTargets(const std::vector<Target> &target, cv::Mat &debug);
        void debugSignalContour(const Contour &contour, cv::Mat &debug);

        MarkerDetector(const MarkerDetectorConfig &cfg) :
            _cfg(cfg),
            reader(cfg) {
        }
    };
}

#endif /* MARKERDETECTOR_H_ */
