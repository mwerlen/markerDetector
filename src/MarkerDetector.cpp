#include "MarkerDetector.h"
#include "SignalReader.h"

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
    void MarkerDetector::detectAndMeasure(const cv::Mat &image, std::vector<Target> &targets, cv::Mat &debug) {
       
        // Detect edges
        cv::Mat edges;
        detectEdges(image, edges);

        // transform to contours
        vector<vector<Point>> contours;
        detectContours(edges, contours);

        // Some debugging
        debugContours(contours, debug);
        
        // Convert contours to ellipses
        vector<Ellipse> ellipses;
        contoursToEllipses(contours, ellipses);
        cout << ellipses.size() << " ellipses found" << endl;
        
        // Clustering ellipses
        std::vector<EllipsesCluster> unfilteredClusters;
        unfilteredClusters = clusterEllipses(ellipses);
        
        // Filtering clusters to deduplicate
        std::vector<EllipsesCluster> clusters;
        filterClusters(unfilteredClusters, clusters);

        cout << "Nombre de clusters (dédupliqués) : " << clusters.size() << endl;

        // Detect markers in clusters
        SignalReader *reader = new SignalReader(_cfg);

        // Looping over clusters to identify targets
        for (int i = 0; i < clusters.size(); ++i) {
            EllipsesCluster cluster = clusters[i];

            // Printing cluster for debug
            debugCluster(cluster, debug);
            
            // Get Contour
            Contour contour;
            reader->getSignalContourInsideEllipse(cluster, contour);
                        
            // Printing cluster for debug
            debugSignalContour(contour, debug);
            
            // Get Signal
            vector<float> signal;
            reader->getSignalFromContour(image, contour, signal);
            
            // Detect targets
            reader->getCorrespondingTargets(image, cluster, signal, targets);
        }

        // Some debugging
        debugTargets(targets, debug);
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
     *  Filtering contours by minimum size and returning Ellipses
     *
     */
    void MarkerDetector::contoursToEllipses(const std::vector<Contour> &contours, std::vector<Ellipse> &ellipses) {
        for( int i = 0; i < contours.size(); i++ ) {
            if(contours[i].size() > _cfg.contourFilterMinSize) { 
                ellipses.resize(ellipses.size() + 1);
                ellipses.back() = fitEllipse( Mat(contours[i]) );
            }
        }
    }
    /*
     * This method associate concentric circles into clusters
     * Best match is computed by center proximity and circle radius ratio (as declared in config)
     *
     */
    void MarkerDetector::filterClusters(const std::vector<EllipsesCluster> &clusters, std::vector<EllipsesCluster> &filteredClusters) {
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
                    && norm(iEllipse.center - jEllipse.center) <= iEllipse.size.width * _cfg.ellipseFilterCloseness) {
                    
                    // Testing for the better matching cluster
                    if ((clusters[i].diffH + clusters[i].diffW + clusters[i].distance) >= (clusters[j].diffH + clusters[j].diffW + clusters[j].distance)) {
                        bestMatch = false;
                    }                   
                }
            }
            
            if(bestMatch) {
                filteredClusters.resize(filteredClusters.size() + 1);
                filteredClusters.back() = clusters[i];
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
        ellipse(debug, cluster.inner, Scalar(0,0,255), 6, 8);
        ellipse(debug, cluster.outer, Scalar(0,255,0), 6, 8);
        cout << "Cluster center at " << cluster.center.x;
        cout << " ; " << cluster.center.y << endl;
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

            circle(debug, Point(target.cx, target.cy), 1, Scalar(0,200,200), 3, 8);
            debug.at<Vec3b>(target.cy, target.cx) = 200,200,0;
            putText(debug, std::to_string(target.markerModelId), Point(target.cx,target.cy), FONT_HERSHEY_SIMPLEX, 5, Scalar(200,200,0), 4, 8);
        }
    }
    
    
    void MarkerDetector::debugSignalContour(const Contour &contour, cv::Mat &debug) {
        std::vector<Contour> contours(1);
        contours[0] = contour;
        drawContours(debug, contours, 0, Scalar(200,200,0), 2, 8, vector<Vec4i>(), 0, Point() );
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
