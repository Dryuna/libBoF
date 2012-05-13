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
#include "bagoffeatures.h"
#include "classifiers.h"

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

    bof::BoFBuilder bofProcess;


    getdir("../101_ObjectCategories", categories);
    int numObjects = categories.size()*0.1;


    for(size_t i = 0; i < categories.size()*0.02; ++i)
    {
        std::cout << categories[i] << std::endl;
        imgFiles.clear();
        getdir("../101_ObjectCategories/" + categories[i], imgFiles);
        for(size_t j = 0; j < imgFiles.size()*0.2; ++j)
        {
            std::string imgName = "../101_ObjectCategories/" + categories[i] + "/" + imgFiles[j];
            std::cout << "Processing " + imgName << std::endl;
            inputImg = cv::imread(imgName, 0);
            if(inputImg.empty())
                continue;
            int numFeatures = siftDetector.extract(inputImg, descriptors, keypoints);
            std::cout << "Found " << numFeatures << " SIFT features..." << std::endl;

            bofProcess.addFeatures(descriptors);
        }
    }

    bof::parameters::ClusteringParameters clusterParams;
    clusterParams.numClusters = 1000;
    clusterParams.numPass = 10;

    std::cout << "Building Codex..." << std::endl;
    bofProcess.buildCodex(clusterParams);
    std::cout << "Done." << std::endl;
    //bofProcess.clearFeatures();

    bof::FeatureHistogram hist;
    bof::ml::SVMClassifier svm;

    for(size_t i = 0; i < categories.size()*0.02; ++i)
    {
        std::cout << categories[i] << std::endl;
        imgFiles.clear();
        getdir("../101_ObjectCategories/" + categories[i], imgFiles);
        for(size_t j = 0; j < imgFiles.size()*0.2; ++j)
        {
            std::string imgName = "../101_ObjectCategories/" + categories[i] + "/" + imgFiles[j];
            std::cout << "Processing " + imgName << std::endl;
            inputImg = cv::imread(imgName, 0);
            if(inputImg.empty())
                continue;
            int numFeatures = siftDetector.extract(inputImg, descriptors, keypoints);
            std::cout << "Found " << numFeatures << " SIFT features..." << std::endl;

            bofProcess.getBoF(descriptors, hist, true);
            hist.setLabel(i);
            svm.add(hist);
        }
    }

    cout << "Training Classifier..." << std::endl;
    svm.train();
    cout << "Done." << std::endl;

    int N, hits;
    double result;
    for(size_t i = 0; i < categories.size()*0.02; ++i)
    {
        std::cout << categories[i] << std::endl;
        imgFiles.clear();
        getdir("../101_ObjectCategories/" + categories[i], imgFiles);
        N = imgFiles.size() - imgFiles.size()*0.2;
        hits = 0;
        for(size_t j = imgFiles.size()*0.2; j < imgFiles.size(); ++j)
        {
            std::string imgName = "../101_ObjectCategories/" + categories[i] + "/" + imgFiles[j];
            //std::cout << "Processing " + imgName << std::endl;
            inputImg = cv::imread(imgName, 0);
            if(inputImg.empty())
                continue;
            int numFeatures = siftDetector.extract(inputImg, descriptors, keypoints);
            //std::cout << "Found " << numFeatures << " SIFT features..." << std::endl;

            bofProcess.getBoF(descriptors, hist);
            hist.setLabel(i);
            result = svm.predict(hist);
            if(result == hist.label)
            {
                hits++;
            }
        }

        double accuracy = (double)hits/(double)N * 100.0;
        std::cout << "Accuracy for " << categories[i] << ": " << accuracy << std::endl;
    }



    return 0;
}
