/*******************************************************************************
**
** Copyright (C) 2015-2019 EPFL (Swiss Federal Institute of Technology)
**
** Contact:
**   Dr. Davide A. Cucci, post-doctoral researcher
**   E-mail: davide.cucci@epfl.ch
**
**   Geodetic Engineering Laboratory,
**   1015 Lausanne, Switzerland (www.topo.epfl.ch).
**
**
**
**   This file is part of visiona.
**
**   visiona is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   visiona is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with visiona.  If not, see <http://www.gnu.org/licenses/>.
**
*******************************************************************************/

/*
 * MarkerDetectorConfig.h
 *
 *  Created on: Mar 20, 2015
 *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
 */

#ifndef MARKERDETECTORCONFIG_H_
#define MARKERDETECTORCONFIG_H_

#include <opencv/cv.h>
#include <libconfig.h++>

namespace markerDetector {

    struct MarkerModel {
        
        int id;
        float signalStartsWith;
        std::vector<float> signalModel; // percentage of the flips
        std::vector<bool> codeModel; // code

    };

    struct MarkerDetectorConfig {

        //Debug
        bool debugEllipseCount = false;
        bool debugClusterCount = false;
        bool debugClusterCenter = false;
        bool debugSignalContour = false;
        bool debugClusterEllipses = false;
        bool writeImage = false;
        bool writeSignal = false;

        // Detection
        int CannyBlurKernelSize;
        int CannyLowerThreshold;
        int CannyHigherThreshold;
        float contourFilterMinSize;
        float ellipseFilterCloseness;
        float clusterRatioMaxError;
        float clusterPixelMaxError;
        float markerxCorrThreshold;
        bool disableCheckOnBigEllipses;

        // Targets
        float markerDiameter;
        float markerInnerDiameter;
        int numberOfDots;
        float markerSignalRadiusPercentage;
        float sheetRadiusPercentage;

        std::vector<MarkerModel *> markerModels;

        MarkerDetectorConfig();

        bool loadConfig(const std::string &file);

      protected:

        void fromSettingToMarkers(libconfig::Setting &s, std::vector<MarkerModel *> &markerModels) {
          markerModels.resize(s.getLength());

          for (int i = 0; i < s.getLength(); i++) {
            libconfig::Setting &m = s[i];
            libconfig::Setting &signal = m["signal"];
            libconfig::Setting &code = m["code"];
            
            MarkerModel *markerModel = new MarkerModel();
            m.lookupValue("id",markerModel->id);
            m.lookupValue("signalStartsWith",markerModel->signalStartsWith);

            markerModel->signalModel.resize(signal.getLength());
            for (int j = 0; j < signal.getLength(); j++) {
              markerModel->signalModel[j] = signal[j];
            }

            markerModel->codeModel.resize(code.getLength());
            for (int j = 0; j < code.getLength(); j++) {
              markerModel->codeModel[j] = code[j];
            }

            markerModels[i] = markerModel;
          }
        }
    };

}

#endif /* MARKERDETECTORCONFIG_H_ */
