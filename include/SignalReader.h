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
        void getSignalContourInsideEllipse(const EllipsesCluster& cluster, Contour &contour);
        cv::Point2f evalEllipse(float angle, const Ellipse& ellipse);
        
        void getSignalFromContour(const cv::Mat& image, Contour &contour, std::vector<float> &signal);
        
        void getCorrespondingTargets(const cv::Mat& image, const EllipsesCluster &cluster, std::vector<float> &signal, std::vector<Target> &targets);
        void normalizeSignal(std::vector<float> &sig_in);
        void computeNormalizedxCorr(const std::vector<float> &sig_in, cv::Mat &out, MarkerModel* markerModel);
        void dumpSignal(const std::string id, const std::vector<float> &signal);

        SignalReader(const MarkerDetectorConfig &cfg) :
            _cfg(cfg) {
        }
    };
}

#endif /* MARKERDETECTOR_H_ */
