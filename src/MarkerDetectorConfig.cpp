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
 * MarkerDetectorConfig.cpp
 *
 *  Created on: Mar 20, 2015
 *      Author: Davide A. Cucci (davide.cucci@epfl.ch)
 *
 *  Modified on: Aug 30, 2018
 *      By : Maxime Werlen
 */

#include "MarkerDetectorConfig.h"

#include <libconfig.h++>

using namespace std;
using namespace libconfig;

namespace markerDetector {

MarkerDetectorConfig::MarkerDetectorConfig() {

  CannyBlurKernelSize = 3;
  CannyLowerThreshold = 15;
  CannyHigherThreshold = 75;

  contourFilterMinSize = 17;

  markerxCorrThreshold = 0.60;
  
  ellipseFilterCloseness = 0.01;

}

bool MarkerDetectorConfig::loadConfig(const std::string &file) {

  Config cf;

  try {
    cf.readFile(file.c_str());
  } catch (const FileIOException &fioex) {
    cerr << " * ERROR: I/O error while reading " << file << endl;
    return false;
  } catch (const ParseException &pex) {
    cerr << " * ERROR: malformed cfg file at " << pex.getFile() << ":"
        << pex.getLine() << " - " << pex.getError() << endl;
    return false;
  }

  const Setting &root = cf.getRoot();

  try {
    Setting &debug = root["debug"];
    debug.lookupValue("ellipseCount", debugEllipseCount);
    debug.lookupValue("clusterCount", debugClusterCount);
    debug.lookupValue("clusterCenter", debugClusterCenter);
    debug.lookupValue("signalContour", debugSignalContour);
    debug.lookupValue("clusterEllipses", debugClusterEllipses);
    debug.lookupValue("writeImage", writeImage);
    debug.lookupValue("writeSignal", writeSignal);
  
    Setting &detection = root["detection"];
    detection.lookupValue("CannyBlurKernelSize", CannyBlurKernelSize);
    detection.lookupValue("CannyLowerThreshold", CannyLowerThreshold);
    detection.lookupValue("CannyHigherThreshold", CannyHigherThreshold);
    detection.lookupValue("contourFilterMinSize", contourFilterMinSize);
    detection.lookupValue("ellipseFilterCloseness", ellipseFilterCloseness);
    detection.lookupValue("clusterRatioMaxError", clusterRatioMaxError);
    detection.lookupValue("clusterPixelMaxError", clusterPixelMaxError);
    detection.lookupValue("markerxCorrThreshold", markerxCorrThreshold);
    detection.lookupValue("disableCheckOnBigEllipses", disableCheckOnBigEllipses);
    
    Setting &targets = root["targets"];

    targets.lookupValue("d", markerDiameter);
    targets.lookupValue("id", markerInnerDiameter);
    targets.lookupValue("numberOfDots", numberOfDots);
    targets.lookupValue("signalRadiusPercentage", markerSignalRadiusPercentage);
    targets.lookupValue("sheetRadiusPercentage", sheetRadiusPercentage);
    

    fromSettingToMarkers(targets["targetModels"], markerModels);

  } catch (SettingNotFoundException &e) {
    cerr << " * ERROR: \"" << e.getPath() << "\" not defined." << endl;
    return false;
  }

  return true;

}

}
