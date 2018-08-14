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

        for (int i = 0; i < clusters.size(); ++i) {
            cout << i << " - " << clusters[i].center.x;
            cout << " - " << clusters[i].center.y << endl;
        }
        
        // Printing clusters for debug
        debugClusters(clusters, debug);

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


    void MarkerDetector::debugClusters(const std::vector<EllipsesCluster> &clusters, cv::Mat &debug) {
        for (int i = 0; i < clusters.size(); i++) {
            ellipse(debug, clusters[i].inner, Scalar(0,0,255), 6, 8);
            ellipse(debug, clusters[i].outer, Scalar(0,255,0), 6, 8);
        }
    }

    void MarkerDetector::debugContours(const std::vector<Contour> &contours, cv::Mat &debug) {
        drawContours(debug, contours, -1, Scalar(255,0,0), 2, 8, vector<Vec4i>(), 0, Point() );
    }
}
