#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "imagefeatures.h"

int getdir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

int main()
{
    std::vector<std::string> categories;
    std::vector<std::string> imgFiles;

    bof::parameters::SIFTParameters siftParams;
    bof::features::SIFTFeatures siftDetector(siftParams);
    cv::Mat inputImg;
    std::vector<bof::FeatureVector> descriptors;
    std::vector<cv::KeyPoint> keypoints;


    getdir("../101_ObjectCategories", categories);
    for(size_t i = 0; i < categories.size(); ++i)
    {
        std::cout << categories[i] << std::endl;
        imgFiles.clear();
        getdir("../101_ObjectCategories/" + categories[i], imgFiles);
        for(size_t j = 0; j < imgFiles.size(); ++j)
        {
            std::string imgName = "../101_ObjectCategories/" + categories[i] + "/" + imgFiles[j];
            std::cout << "Processing " + imgName << std::endl;
            inputImg = cv::imread(imgName, 0);
            if(inputImg.empty())
                continue;
            int numFeatures = siftDetector.extract(inputImg, descriptors, keypoints);
            std::cout << "Found " << numFeatures << " SIFT features..." << std::endl;
        }
    }



    return 0;
}
