#include "MarkerDetector.h"
#include "SignalReader.h"

#include <vector>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

namespace markerDetector {

    /*
     * Detect targets and measure center
     */
    void MarkerDetector::detectAndMeasure(const cv::Mat &image, std::vector<Target> &targets, cv::Mat &debug) {
       
        // On passe l'image en niveaux de gris
        cv::Mat image_gray;
        cvtColor(image, image_gray, CV_BGR2GRAY);
        cv::Mat tmp;
        
        // Detect edges
        cv::Mat edges;
        detectEdges(image_gray, edges);

        // transform to contours
        vector<vector<Point>> contours;
        detectContours(edges, contours);

        // Some debugging
        debugContours(contours, debug);
        
        // Convert contours to ellipses
        vector<Ellipse> ellipses;
        contoursToEllipses(contours, ellipses);
        
        if (_cfg.debugEllipseCount) {
            cout << ellipses.size() << " ellipses found" << endl;
        }
        
        // Clustering ellipses
        std::vector<EllipsesCluster> unfilteredClusters;
        unfilteredClusters = clusterEllipses(ellipses);
        
        // Filtering clusters to deduplicate
        std::vector<EllipsesCluster> clusters;
        filterClusters(unfilteredClusters, clusters, image_gray);

        if (_cfg.debugClusterCount) {
            cout << "Nombre de clusters (dédupliqués) : " << clusters.size() << endl;
        }


        // Looping over clusters to identify targets
        for (int i = 0; i < clusters.size(); ++i) {
            EllipsesCluster cluster = clusters[i];

            // Printing cluster for debug
            debugCluster(cluster, debug);
            
            // Get Contour
            Contour contour;
            reader.getSignalContourInsideEllipse(cluster.inner, contour, _cfg.markerSignalRadiusPercentage);
                        
            // Printing cluster for debug
            debugSignalContour(contour, debug);
            
            // Get Signal
            vector<float> signal;
            reader.getSignalFromContour(image_gray, contour, signal);
            
            // Detect targets
            reader.getCorrespondingTargets(image, cluster, signal, targets);
        }

        // Some debugging
        debugTargets(targets, debug);
    }



    /*
     * Edge detection is done with Canny from openCV
     *
     */
    void MarkerDetector::detectEdges(const cv::Mat& image, cv::Mat& edges) {
       
        cv::Mat tmp;

        // with canny
        if (_cfg.CannyBlurKernelSize > 0) {
            blur(image, tmp, Size(_cfg.CannyBlurKernelSize, _cfg.CannyBlurKernelSize));
            Canny(tmp, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
        } else {
            Canny(image, edges, _cfg.CannyLowerThreshold, _cfg.CannyHigherThreshold, 3, true);
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
     *  Filtering contours by minimum size and returning Ellipses
     *
     */
    void MarkerDetector::contoursToEllipses(const std::vector<Contour> &contours, std::vector<Ellipse> &ellipses) {
        for( int i = 0; i < contours.size(); i++ ) {
            if(contours[i].size() > _cfg.contourFilterMinSize) { 
                Ellipse ellipse = fitEllipse(Mat(contours[i]));
                if ((ellipse.size.height <= 2 * ellipse.size.width) && (ellipse.size.width <= 2 * ellipse.size.height)) {
                    ellipses.resize(ellipses.size() + 1);
                    ellipses.back() = fitEllipse( Mat(contours[i]) );
                }
            }
        }
    }
    /*
     * This method associate concentric circles into clusters
     * Best match is computed by center proximity and circle radius ratio (as declared in config)
     *
     */
    void MarkerDetector::filterClusters(const std::vector<EllipsesCluster> &clusters, std::vector<EllipsesCluster> &filteredClusters, const cv::Mat &image) {

        for (int i = 0; i < clusters.size(); ++i) {
            bool bestMatch = true;
            Ellipse iEllipse = clusters[i].outer;
        
            for (int j = 0; j < clusters.size(); ++j) {
                if ( i == j ) { 
                    continue;
                 }

                Ellipse jEllipse = clusters[j].outer;
                // Testing if cluters[i] is looking like clusters[j]
                if (fabs(iEllipse.size.height - jEllipse.size.height) <= (iEllipse.size.height * _cfg.ellipseFilterCloseness)
                    && fabs(iEllipse.size.width - jEllipse.size.width)  <= (iEllipse.size.width * _cfg.ellipseFilterCloseness)
                    && norm(iEllipse.center - jEllipse.center) <= iEllipse.size.width) {
                    
                    // Testing for the better matching cluster
                    if ((clusters[i].diffH + clusters[i].diffW + clusters[i].distance) >= (clusters[j].diffH + clusters[j].diffW + clusters[j].distance)) {
                        bestMatch = false;
                    }                   
                }
            }
            
            if(bestMatch) {
                if ((_cfg.disableCheckOnBigEllipses && iEllipse.size.height > 300)
                    || reader.checkCluster(image, clusters[i])) {
                    filteredClusters.resize(filteredClusters.size() + 1);
                    filteredClusters.back() = clusters[i];
                }
            }
        }
    }


    /*
     * This method associate concentric circles into clusters
     * Best match is computed by center proximity and circle radius ratio (as declared in config)
     *
     */
    std::vector<EllipsesCluster> MarkerDetector::clusterEllipses(const std::vector<Ellipse> &ellipses) {

      std::vector<EllipsesCluster> clusters;
      // for each circle, do a cluster with the best matching inner circle, if found
      float radiusRatio = _cfg.markerInnerDiameter / _cfg.markerDiameter;

      for (int i = 0; i < ellipses.size(); ++i) {

        int bestMatch = -1;
        float bestDiffW = ellipses[i].size.width * _cfg.clusterRatioMaxError; // x% error max
        float bestDiffH = ellipses[i].size.height * _cfg.clusterRatioMaxError; // x% error max

        for (int j = 0; j < ellipses.size(); ++j) {
          if ( ellipses[i].size.height > ellipses[j].size.height        // Ellipse i is higher than j
            && ellipses[i].size.width > ellipses[j].size.width          // Ellipse i is wider than j
            && norm(ellipses[i].center - ellipses[j].center) < max((ellipses[i].size.width + ellipses[i].size.height) * _cfg.clusterRatioMaxError,_cfg.clusterPixelMaxError) // centers are not too far
            && norm(ellipses[i].center - ellipses[j].center) < min(ellipses[j].size.width, ellipses[j].size.height)) { // Center of i is inside j
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
          clusters.back().outer = ellipses[i];
          clusters.back().inner = ellipses[bestMatch];
          clusters.back().diffW = bestDiffW;
          clusters.back().diffH = bestDiffH;
          clusters.back().distance = norm(ellipses[i].center - ellipses[bestMatch].center);
        }
      }
      return clusters;
    }



    void MarkerDetector::debugCluster(const EllipsesCluster &cluster, cv::Mat &debug) {
        if (_cfg.debugClusterEllipses) {
            ellipse(debug, cluster.inner, Scalar(0,0,255), 6, 8);
            circle(debug, Point(cluster.inner.center.x, cluster.inner.center.y), 1, Scalar(0,0,255), 6, 8);
            ellipse(debug, cluster.outer, Scalar(0,255,0), 6, 8);
            circle(debug, Point(cluster.outer.center.x, cluster.outer.center.y), 1, Scalar(0,255,0), 6, 8);
        }
        if (_cfg.debugClusterCenter) {
            cout << "---" << endl;
            cout << "Cluster center at " << cluster.center.x;
            cout << " ; " << cluster.center.y << endl;
        }
    }

    void MarkerDetector::debugContours(const std::vector<Contour> &contours, cv::Mat &debug) {
        drawContours(debug, contours, -1, Scalar(255,0,0), 2, 8, vector<Vec4i>(), 0, Point() );
    }

    void MarkerDetector::debugTargets(const std::vector<Target> &targets, cv::Mat &debug) {
        for (int i = 0; i < targets.size(); i++) {
            Target target = targets[i];
            cout << "Detected target " << target.markerModelId << " at ";
            cout << fixed << setprecision(6) << target.cx << ";";
            cout << fixed << setprecision(6) << target.cy;
            cout << " with score " << target.correlationScore << endl;

            debug.at<Vec3b>(target.cy, target.cx) = 200,200,0;
            putText(debug, std::to_string(target.markerModelId), Point(target.cx,target.cy), FONT_HERSHEY_SIMPLEX, 5, Scalar(200,200,0), 4, 8);
        }
    }
    
    
    void MarkerDetector::debugSignalContour(const Contour &contour, cv::Mat &debug) {
        if (_cfg.debugSignalContour) {
            std::vector<Contour> contours(1);
            contours[0] = contour;
            drawContours(debug, contours, 0, Scalar(200,200,0), 2, 8, vector<Vec4i>(), 0, Point() );

            cout << "Signal length : ";
            cout << fixed << setprecision(6) << contour.size();
            cout << endl;
        }
    }
    /*
    // Pour debugger les X/Y
    void MarkerDetector::debugSignalContour(const Contour &contour, cv::Mat &debug) {
        for (int i = 0; i < contour.size(); ++i) {
            Point2i px = contour[i];
            debug.at<Vec3b>(px.y, px.x) = 200,200,0;
        }
    }
    */
}
