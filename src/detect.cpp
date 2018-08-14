#include <fstream>
#include <iostream>
#include <vector>

#include <dirent.h>
#include <unistd.h>
#include <getopt.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <libconfig.h++>

#include "markerDetector.h"

using namespace std;
using namespace cv;
using namespace libconfig;
using namespace markerDetector;

#define OPTPATHSET 1
#define OPTEXTSET 2
#define OPTCFGFILESET 4
#define OPTDEBUG 8
#define OPTPREFIXSET 16
#define OPTUSESEEDPOINTS 32 

int main(int argc, char *argv[]) {

  // --------------------- parse command line arguments ------------------------

  static struct option long_options[] = {
      { "path", required_argument, 0, 'p' },
      { "ext", required_argument, 0, 'e' },
      { "config", required_argument, 0, 'c' },
      { "debug", no_argument, 0, 'd' },
      { "prefix", required_argument, 0, 'f' },
      { 0, 0, 0, 0 }
  };

  unsigned int optionflag = 0;
  char *imagePath, *imgext, *configpath, *prefix, *detectioncfgpath, *seedpointspath;

  opterr = 0;
  int c;
  while ((c = getopt_long_only(argc, argv, "", long_options, NULL)) != -1) {
    switch (c) {
    case 'p':
      if (optarg[strlen(optarg) - 1] == '/') {
        optarg[strlen(optarg) - 1] = 0;
      }
      imagePath = optarg;
      optionflag |= OPTPATHSET;
      break;
    case 'e':
      imgext = optarg;
      optionflag |= OPTEXTSET;
      break;
    case 'c':
      configpath = optarg;
      optionflag |= OPTCFGFILESET;
      break;
    case 'd':
      optionflag |= OPTDEBUG;
      break;
    case 'f':
      prefix = optarg;
      optionflag |= OPTPREFIXSET;
      break;
    case '?':
      cerr << " * ERROR: unknown option or missing argument" << endl;
      return 1;

    default:
      abort();
    }
  }

  if ((optionflag & OPTPATHSET) == 0) {
    cerr << " * ERROR: image path not specified (-p)" << endl;
    return 1;
  }
  if ((optionflag & OPTEXTSET) == 0) {
    cerr << " * ERROR: image extension not specified (-e)" << endl;
    return 1;
  }
  if ((optionflag & OPTCFGFILESET) == 0) {
    cerr << " * ERROR: config file not specified (-c)" << endl;
    return 1;
  }

  DIR *dir;
  if ((dir = opendir(imagePath)) == NULL) {
    cerr << " * ERROR: could not open image folder" << endl;
    return 1;
  }

  // --------------------- configuration ---------------------------------------

  // loading config
  MarkerDetectorConfig cfg;
  if (!cfg.loadConfig(configpath)) {
    return 1;
  }

  MarkerDetector *detector = new MarkerDetector(cfg);

  // --------------------- generating image list -------------------------------

  vector<string> images;

  struct dirent *file;

  while ((file = readdir(dir)) != NULL) {
    if (strcmp(file->d_name + strlen(file->d_name) - 3, imgext) == 0) {

      string fn_prefix = "_";
      if (optionflag & OPTPREFIXSET) {
        fn_prefix = string(prefix);
      }

      string fname(file->d_name);

      int sep = fname.find(fn_prefix);
      if (sep != string::npos) {
        images.push_back(file->d_name);
      }
    }
  }

  closedir(dir);

  // --------------------- prepare output files --------------------------------
  ofstream *output = new ofstream("output.csv");
  *output << "file;target_id;x;y" << endl;

  cout << images.size() << " file(s) found." << endl;

  // --------------------- process every image ---------------------------------
  for (int i = 0; i < images.size(); i++) {
    string filename = images[i];

    cout << "----------------------------------------------------------" << endl;
    cout << filename << " ..." << endl;
    

    string imgName = imagePath + string("/") + filename;
    string debugImgName = string("debug/") + filename;

    Mat raw = imread(imgName, CV_LOAD_IMAGE_COLOR);
    Mat debug = raw.clone();

    // Launch detection
    vector<Target> returnedValues = detector->detectAndMeasure(raw, debug);


    imwrite(debugImgName, debug);
  
    // Log
    for (auto targetIt = returnedValues.begin(); targetIt != returnedValues.end(); ++targetIt) {
      Target &target = *targetIt;

      if (target.measured) {
        *output << filename.substr(0, filename.length() - 4) << ";";
        *output << target.markerModel->id << ";";
        *output << fixed << setprecision(6) << target.cx << ";";
        *output << fixed << setprecision(6) << target.cy << endl;
        output->flush();

        cout << "Detected target " << target.markerModel->id << " at ";
        cout << fixed << setprecision(6) << target.cx << ";";
        cout << fixed << setprecision(6) << target.cy << " - ";
        cout << fixed << setprecision(6) << target.outer.center.x << ";";
        cout << fixed << setprecision(6) << target.outer.center.y << endl;
      } else {
        cout << "Unable to measure :(" << endl;
      }
    }
  }

  return 0;
}
