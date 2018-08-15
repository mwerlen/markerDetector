#include "SignalReader.h"
#include "MarkerDetector.h"

#include <libconfig.h++>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace libconfig;
using namespace cv;

namespace markerDetector {

    /*
     * Get signal along an ellipse (given ellipse)
     * 
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    void SignalReader::getSignalContourInsideEllipse(const EllipsesCluster& cluster, Contour &contour) {

        const Ellipse &ellipse = cluster.outer;
        
        // Computing ellipse perimeter in pixels - perimeter = PI * sqrt(2*(a²+b²))
        int N = ceil(sqrt(2 * (pow(ellipse.size.width, 2.0) + pow(ellipse.size.height, 2.0))) / 2);
        float increment = 2 * M_PI / N;

        //Initializing signal
        contour.clear();
        contour.reserve(N);
        float theta = 0.0;

        for (int i = 0; i < N; ++i, theta += increment) {
            Point2f px = evalEllipse(theta, ellipse);
            contour.push_back(px);
        }
    }
    

    /*
     * Get X/Y coordinate of a point on an ellipse (determined by center, a & b radiuses) at a specified phi angle
     *
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    Point2f SignalReader::evalEllipse(float angle, const Ellipse &ellipse) {
        const Point2f& center = ellipse.center;
        float a = ellipse.size.width * _cfg.markerSignalRadiusPercentage / 2.0;
        float b = ellipse.size.height * _cfg.markerSignalRadiusPercentage / 2.0;
        float phi = ellipse.angle * M_PI / 180.0;

        float offset = 0.0;
        if (a < b) {
            offset = M_PI / 2.0;
            std::swap(a, b);
        }

        Point2f ret;

        ret.x = center.x + a * cos(angle - phi + offset - M_PI) * cos(phi + offset) - b * sin(angle - phi + offset - M_PI) * sin(phi + offset);
        ret.y = center.y + a * cos(angle - phi + offset - M_PI) * sin(phi + offset) + b * sin(angle - phi + offset - M_PI) * cos(phi + offset);

        return ret;
    }


    /*
     * Get signal along an ellipse (given ellipse)
     * 
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    void SignalReader::getSignalFromContour(const Mat& image, Contour &contour, std::vector<float> &signal) {

        signal.clear();
        signal.reserve(contour.size());

        for (int i = 0; i < contour.size(); ++i) {
            Point2i px = contour[i];
            Scalar intensity = image.at<uchar>(px.y, px.x);
            signal.push_back(intensity[0]);
        }
    }

    /*
     * For each cluster, this method compares signal with reference signal
     * Cluster having maximal correlation score selected and theta value (angle between ref and signal is computed)
     * returned values are selectedCluster and theta pointers
     *
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    void SignalReader::getCorrespondingTargets(const Mat& image, const EllipsesCluster &cluster, std::vector<float> &signal, std::vector<Target> &targets) {

        float maxCorrelation = _cfg.markerxCorrThreshold;

        int selectedMarkerModelId;
        bool found = false;

        normalizeSignal(signal);
       

        for (int markerModelId = 0; markerModelId < _cfg.markerModels.size(); markerModelId++) {
            MarkerModel * markerModel = _cfg.markerModels[markerModelId];
            Mat correlationMatrix;
            computeNormalizedxCorr(signal, correlationMatrix, markerModel);
          
            double min, max;
            Point2i minLocation, maxLocation;
          
            minMaxLoc(correlationMatrix, &min, &max, &minLocation, &maxLocation);
                       
            if (max > maxCorrelation) {
                maxCorrelation = max;
                selectedMarkerModelId = markerModel->id;
                found = true;
            }
        }
    
        if (found) {
            targets.reserve(targets.size() + 1);
            targets.push_back(Target());
            targets.back().outer = cluster.outer;
            targets.back().inner = cluster.inner;
            targets.back().markerModelId = selectedMarkerModelId;
            targets.back().cx = cluster.outer.center.x;
            targets.back().cy = cluster.outer.center.y;
            targets.back().correlationScore = maxCorrelation;
        }
    }



    /*
     * Normalize signal between -1 and 1 after computing signal min and max value
     *
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    void SignalReader::normalizeSignal(std::vector<float> &sig_in) {
        // depolarize
        Mat sig(1, sig_in.size(), CV_32FC1, sig_in.data());
        
        float u = mean(sig)[0];

        sig -= u;

        float max = -std::numeric_limits<float>::infinity();
        float min = std::numeric_limits<float>::infinity();

        for (unsigned int k = 0; k < sig_in.size(); k++) {
            if (sig_in[k] > max) {
                max = sig_in[k];
            } else if (sig_in[k] < min) {
                min = sig_in[k];
            }
        }

        if (min != max) {
            for (unsigned int k = 0; k < sig_in.size(); k++) {
                sig_in[k] = -1.0 + (sig_in[k] - min) * 2.0 / (max - min);
            }
        }
    }

    /*
     * computeNormalizedxCorr compares reference signal with picture's target signal
     * Give back out matrice with diff between ref and signal.
     * 
     *  Created on: Mar 20, 2015
     *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
     *      See https://github.com/DavideACucci/visiona
     */
    void SignalReader::computeNormalizedxCorr(const std::vector<float>& sig_in, Mat &out, MarkerModel* markerModel) {

        // prepare signals
        float raw[2 * sig_in.size()];

        for (unsigned int n = 0; n < 2; n++) {
            unsigned int segment = 0;
            float val = markerModel->signalStartsWith;

            for (unsigned int k = 0; k < sig_in.size(); k++) {
                if (segment < markerModel->signalModel.size()) {
                    // test if I have to advance
                    if (k > markerModel->signalModel[segment] * sig_in.size()) {
                        segment++;
                        val = -val;
                    }
                }
                raw[k + n * sig_in.size()] = val;
            }
        }

        Mat ref(1, 2 * sig_in.size(), CV_32FC1, raw);
        Mat sig(1, sig_in.size(), CV_32FC1, const_cast<float *>(sig_in.data()));

        // compute cross correlation
        matchTemplate(ref, sig, out, CV_TM_CCORR_NORMED);
    }
}
