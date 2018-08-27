#ifndef TARGET_H_
#define TARGET_H_

#include <vector>
#include "Structures.h"


namespace markerDetector {

    /**
     * A target
     * includes temporary and final results of the detect and measure process
     */

    class Target {
      public:
        
        // inner and outer target ellipse approximations
        Ellipse inner, outer;
        
        // Center in image
        double cx, cy;
        
        // MarkerModel detected
        int markerModelId;

        //Correlation score
        float correlationScore;

        Target(){};
    };
}

#endif /* TARGET_H_ */
