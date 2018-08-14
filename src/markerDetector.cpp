#include "markerDetector.h"

#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

namespace markerDetector {

    /*
     * Detect targets and measure center
     */
    std::vector<Target> MarkerDetector::detectAndMeasure(const cv::Mat &image, cv::Mat &debug) {

        std::vector<Target> targets;
        
        cv::Mat edges;
        detectEdges(image, edges);

        vector<vector<Point>> contours;
        detectContours(edges, contours);

        // Find the rotated rectangles and ellipses for each contour
        vector<RotatedRect> minRect (contours.size());
        vector<Ellipse> ellipses;
        vector<Ellipse> filteredEllipses;

        for( int i = 0; i < contours.size(); i++ ) {
            minRect[i] = minAreaRect(Mat(contours[i]));
            if(contours[i].size() > _cfg.contourFilterMinSize) { 
                ellipses.resize(ellipses.size() + 1);
                ellipses.back() = fitEllipse( Mat(contours[i]) );
                drawContours(debug, contours, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
            }
        }
        
        filterEllipses(ellipses, filteredEllipses);
        
        cout << filteredEllipses.size() << " ellipses found" << endl;
        
        std::vector<EllipsesCluster> clusters;
        clusters = clusterEllipses(filteredEllipses, debug);

        cout << "Nombre de clusters : " << clusters.size() << endl;

        for (int i = 0; i < clusters.size(); ++i) {
            cout << i << " - " << clusters[i].center.x;
            cout << " - " << clusters[i].center.y << endl;
        }

        return targets;
    }



    /*
     * Edge detection is done with Canny from openCV
     *
     */
    void MarkerDetector::detectEdges(const cv::Mat& image, cv::Mat& edges) {
        
        // On passe l'image en niveaux de gris
        Mat image_gray;
        cvtColor(image, image_gray, CV_BGR2GRAY);
        cv::Mat tmp;

        // with canny
        if (_cfg.CannyBlurKernelSize > 0) {
            blur(image_gray, tmp, Size(_cfg.CannyBlurKernelSize, _cfg.CannyBlurKernelSize));
            Canny(tmp, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
        } else {
            Canny(image_gray, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
        }
    }
    
    /*
    void MarkerDetector::detectEdges(const cv::Mat& image, cv::Mat& edges) {
        // Detect edges using Threshold
        Mat src_gray;
        cvtColor(image, src_gray, CV_BGR2GRAY);
        threshold(src_gray, edges, 128, 255, THRESH_BINARY);
    }*/


    /*
     * Contours detection is done with openCV method
     *
     */
    void MarkerDetector::detectContours(const cv::Mat &edges, vector<vector<Point>> &ctrs) {
      ctrs.clear();
      findContours(edges, ctrs, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));
    }
    
    /*
     * This method associate concentric circles into clusters
     * Best match is computed by center proximity and circle radius ratio (as declared in config)
     *
     */
    void MarkerDetector::filterEllipses(const std::vector<Ellipse> &ellipses, std::vector<Ellipse> &filteredEllipses) {
        for (int i = 0; i < ellipses.size(); ++i) {
            bool alreadyExists = false;
        
            for (int j = 0; j < filteredEllipses.size(); ++j) {
                // Testing if ellipse[i] is looking like filteredEllipses[j]
                if ( fabs(ellipses[i].size.height - filteredEllipses[j].size.height) < (ellipses[i].size.height * _cfg.ellipseFilterCloseness)
                    && fabs(ellipses[i].size.width - filteredEllipses[j].size.width)  < (ellipses[i].size.width * _cfg.ellipseFilterCloseness)
                    && norm(ellipses[i].center - filteredEllipses[j].center) < ellipses[i].size.width * _cfg.ellipseFilterCloseness) {
                    alreadyExists = true;
                    
                    // TODO MWE : keep best ellipse, not the first !
                }
            }
            
            if(!alreadyExists) {
                filteredEllipses.resize(filteredEllipses.size() + 1);
                filteredEllipses.back() = ellipses[i];
            }
        }
    }


    /*
     * This method associate concentric circles into clusters
     * Best match is computed by center proximity and circle radius ratio (as declared in config)
     *
     */
    std::vector<EllipsesCluster> MarkerDetector::clusterEllipses(const std::vector<Ellipse> &ellipses, cv::Mat &debug) {

      std::vector<EllipsesCluster> clusters;
      // for each circle, do a cluster with the best matching inner circle, if found
      float radiusRatio = _cfg.markerInnerDiameter / _cfg.markerDiameter;

      for (int i = 0; i < ellipses.size(); ++i) {

        int bestMatch = -1;
        float bestDiffW = ellipses[i].size.width * 0.05; // 5% error max
        float bestDiffH = ellipses[i].size.height * 0.05; // 5% error max

        for (int j = 0; j < ellipses.size(); ++j) {
          if ( ellipses[i].size.height > ellipses[j].size.height
            && ellipses[i].size.width > ellipses[j].size.width
            && norm(ellipses[i].center - ellipses[j].center) < ellipses[i].size.width * 0.1) {
            float curDiffH = fabs(ellipses[i].size.height * radiusRatio - ellipses[j].size.height);
            float curDiffW = fabs(ellipses[i].size.width * radiusRatio - ellipses[j].size.width);

            if ((curDiffW < bestDiffW) && (curDiffH < bestDiffH)) {
              bestMatch = j;
              bestDiffW = curDiffW;
              bestDiffW = curDiffH; 
            }
          }
        }

        if (bestMatch != -1) {
          clusters.resize(clusters.size() + 1);
          clusters.back().center = ellipses[i].center;
          clusters.back().rep = i;
          clusters.back().ellipsesIds.push_back(i);
          clusters.back().ellipsesIds.push_back(bestMatch);
          
          ellipse(debug, ellipses[i], Scalar(0,0,255), 4, 8 );
          ellipse(debug, ellipses[bestMatch], Scalar(0,255,0), 4, 8 );
        }
      }
      return clusters;
    }
}
