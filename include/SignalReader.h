#ifndef SIGNALREADER_H_
#define SIGNALREADER_H_

#include <vector>
#include <memory>

#include <opencv/cv.h>

#include "MarkerDetectorConfig.h"
#include "MarkerDetector.h"

namespace markerDetector {
    class SignalReader {
        protected:
            MarkerDetectorConfig _cfg;

        public:
        void getSignalContourInsideEllipse(const Ellipse &ellipse, Contour &contour, float signalRadiusPercentage);
        cv::Point2f evalEllipse(float angle, const Ellipse &ellipse, float signalRadiusPercentage);
        bool checkCluster(const cv::Mat &image, const EllipsesCluster cluster);

        void getSignalFromContour(const cv::Mat& image, Contour &contour, std::vector<float> &signal);
        
        void getCorrespondingTargets(const cv::Mat& image, const EllipsesCluster &cluster, std::vector<float> &signal, std::vector<Target> &targets);
        void normalizeSignal(std::vector<float> &sig_in);
        void flattenSignal(std::vector<float> &sig_in);
        void smoothSignal(std::vector<float> &sig_in, std::vector<float> &sig_out);
        float computeSmoothedValue(std::vector<float> &signal, int id);
        void computeCenters(std::vector<float> &signal, std::vector<float> &centersOfDots);
        float computeOffset(int signalSize, std::vector<float> &centersOfDots);
        void getCode(int signalSize, std::vector<float> centers, float offset, bool code[]); 
        bool checkForBlackParts(std::vector<float> &signal, float offset);
        void computeNormalizedxCorr(const std::vector<float> &sig_in, cv::Mat &out, MarkerModel* markerModel);
        
        void dumpSignal(const std::vector<float> &signal, cv::Mat &debug, cv::Vec3b color);
        void dumpCenters(const std::vector<float> &centers, cv::Mat &debug, cv::Scalar color);
        void dumpOffset(const int offset, cv::Mat &debug, cv::Scalar color);
        void dumpCode(const bool code[]);

        SignalReader(const MarkerDetectorConfig &cfg) :
            _cfg(cfg) {
        }
    };
}

#endif /* MARKERDETECTOR_H_ */
