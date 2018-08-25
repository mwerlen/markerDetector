#include "SignalReader.h"
#include "MarkerDetector.h"

#include <libconfig.h++>
#include <fstream>
#include <iostream>
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

        const Ellipse &ellipse = cluster.inner;
        
        // Computing ellipse perimeter in pixels - perimeter = PI * sqrt(2*(a²+b²))
        int N = ceil(
                    sqrt(2 * (
                        pow(ellipse.size.width  * _cfg.markerSignalRadiusPercentage, 2.0)
                        + pow(ellipse.size.height * _cfg.markerSignalRadiusPercentage, 2.0)
                        )
                    ) / 2
                );
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
        Mat image_gray;
        cvtColor(image, image_gray, CV_BGR2GRAY);

        signal.clear();
        signal.reserve(contour.size());

        for (int i = 0; i < contour.size(); ++i) {
            Point2i px = contour[i];
            Scalar intensity = image_gray.at<uchar>(px.y, px.x); // Sisi, c'est bien y puis x
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

        Mat debug(220,signal.size(),CV_8UC3,Scalar(255,255,255));
        
        float maxCorrelation = _cfg.markerxCorrThreshold;

        int selectedMarkerModelId;
        bool found = false;

        normalizeSignal(signal);
        
        // Prepare signal
        vector<float> smoothedSignal;
        smoothSignal(signal, smoothedSignal);
        flattenSignal(smoothedSignal);

        // Detect dots centers
        vector<float> centers;
        computeCenters(smoothedSignal, centers);

        // Compute offset
        float offset;
        offset = computeOffset(signal.size(), centers);
    
        // Some debugs
        dumpCenters(centers, debug, Scalar(0,255,255));
        dumpOffset(offset, debug, Scalar(0,0,255));
        dumpSignal(smoothedSignal, debug, Vec3b(0,255,0));
        dumpSignal(signal, debug, Vec3b(255,0,0));
        imwrite("debug/signal-"+std::to_string(cluster.center.x)+".jpg", debug);

        // Compute code
        bool code[_cfg.numberOfDots];
        getCode(signal.size(), centers, offset, code);
        dumpCode(code);

        // Check black dots are really black !
        // TODO

        for (int markerModelId = 0; markerModelId < _cfg.markerModels.size(); markerModelId++) {
            MarkerModel * markerModel = _cfg.markerModels[markerModelId];
            Mat correlationMatrix;
            computeNormalizedxCorr(smoothedSignal, correlationMatrix, markerModel);
          
            double min, max;
            Point2i minLocation, maxLocation;
          
            minMaxLoc(correlationMatrix, &min, &max, &minLocation, &maxLocation);
            
            if(max > _cfg.markerxCorrThreshold) {
                cout << "Matching with model " << markerModel->id << " - " << max << " -" << maxLocation.x << "/" << correlationMatrix.cols << endl;
            }
            
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
            targets.back().cx = cluster.inner.center.x;
            targets.back().cy = cluster.inner.center.y;
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
                /*if(sig_in[k]<0) {
                    sig_in[k] = -1.0;
                } else {
                    sig_in[k] = 1.0;
                }*/
            }
        }
    }

    void SignalReader::flattenSignal(std::vector<float> &sig_in) {
        
        for (unsigned int k = 0; k < sig_in.size(); k++) {
            if(sig_in[k]<0) {
                sig_in[k] = -1.0;
            } else {
                sig_in[k] = 1.0;
            }
        }
    }


    void SignalReader::smoothSignal(std::vector<float> &sig_in, std::vector<float> &sig_out) {

        // Smoothed signal preparation
        sig_out.clear();
        sig_out.resize(sig_in.size());

        for (int i = 0; i < sig_in.size(); i++){
            sig_out[i] = computeSmoothedValue(sig_in, i);
        }

        cout << "Smoothed signal with " << sig_out.size() << " points" << endl;
    }


    float SignalReader::computeSmoothedValue(std::vector<float> &signal, int id) {
        
        // Compute smoother size
        int smootherSize = signal.size() / _cfg.numberOfDots / 8;

        int min = floor(id - (smootherSize/2));
        int max = ceil(id + (smootherSize/2));

        float value = 0;
        float totalCoeff = 0;
        for (int i = min; i < max; i ++) {
            int coeff = (smootherSize/2) - abs(id-i);
            int index = (i + signal.size()) % signal.size();
            value += signal[index] * coeff;
            totalCoeff += coeff;
        }

        return value / totalCoeff;
    }


    /*
     *  This method detects dots (white parts of signal) and computes center
     *  Populate a vector of centers
     */
    void SignalReader::computeCenters(std::vector<float> &signal, std::vector<float> &centersOfDots) {
        
        int startOfDot = -1;
        int endOfDot = -1;
        float lastVal = signal[0];
        float curVal;
        int end = signal.size();
        if (signal[0] > 0) {
            end = signal.size() * ( 1 + ( 1 / _cfg.numberOfDots));
        }
        for (int i = 0; i < end; i++) {
            curVal = signal[i % signal.size()];
            if (lastVal == curVal) {
                continue;
            } else if (lastVal < 0 && curVal > 0) {
                startOfDot = i;
            } else if (lastVal > 0 && curVal < 0) {
                endOfDot = i;
            }

            if (startOfDot != -1 && endOfDot != -1) {
                float center;
                if (startOfDot < endOfDot) {
                   center = (startOfDot + endOfDot) / 2;
                } else {
                    center = (startOfDot + endOfDot + signal.size()) / 2;
                }
                centersOfDots.reserve(centersOfDots.size()+1);
                centersOfDots.push_back(center);
                startOfDot = -1;
                endOfDot = -1;
            }
            lastVal = curVal;
        }
    }

    /*
     *  This method compute the best offset from a set of offset
     *
     */
    float SignalReader::computeOffset(int signalSize, std::vector<float> &centersOfDots) {
        
        if (centersOfDots.size() == 0) {
            return -1;
        } 
        float total = 0;
        float modulo = signalSize / _cfg.numberOfDots;
        for (int i = 0; i < centersOfDots.size(); i++) {
            total += fmod(centersOfDots[i],modulo);
        }
        return total/centersOfDots.size();
    }


    void SignalReader::getCode(int signalSize, std::vector<float> centers, float offset, bool code[]) {
       for (int i = 0; i < _cfg.numberOfDots; i++) {
            int measure = offset + ((signalSize / _cfg.numberOfDots) * i);
            code[i] = false;
            for (int j = 0; j < centers.size(); j++) {
                if (abs(measure - centers[j]) < ((signalSize / _cfg.numberOfDots) / 2 * _cfg.markerxCorrThreshold)) {
                    code[i] = true;
                }
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
        
        //dumpSignal("model_"+std::to_string(markerModel->id), std::vector<float>(raw,raw + sizeof raw / sizeof raw[0]));

        // compute cross correlation
        matchTemplate(ref, sig, out, CV_TM_CCORR_NORMED); //CV_TM_CCORR_NORMED
    }
    
    void SignalReader::dumpSignal(const std::vector<float> &signal, cv::Mat &debug, cv::Vec3b color) {
        for (int i = 0; i < signal.size(); i++) {
            int height = 110 - (signal[i]*100);
            debug.at<Vec3b>(Point(i,height)) = color;
        }
    }

    void SignalReader::dumpCenters(const std::vector<float> &centers, cv::Mat &debug, cv::Scalar color) {
        for (int i = 0; i < centers.size(); i++) {
            line(debug, Point(centers[i],0), Point(centers[i], debug.rows), color, 3);
        }
    }

    void SignalReader::dumpOffset(const int offset, cv::Mat &debug, cv::Scalar color) {
        for (int i = 0; i < _cfg.numberOfDots; i++) {
            int y = ((debug.cols / _cfg.numberOfDots) * i) + offset;
            line(debug, Point(y,0), Point(y, debug.rows), color);
        }
    }

    void SignalReader::dumpCode(const bool code[]) {
        cout << "Code : ";
        for (int i = 0; i < _cfg.numberOfDots; i++) {
            if(code[i]) {
                cout << "0";
            } else {
                cout << "_";
            }
        }
        cout << endl;
    }
}
