#include "structures.h"

#include <assert.h>
#include <math.h>
#include <fstream>

namespace bof
{

////////////////////////////////////////////////////////////////////////
//FeatureVector
////////////////////////////////////////////////////////////////////////
FeatureVector::FeatureVector()
{
    size = 0;
}

FeatureVector::~FeatureVector()
{
    size = 0;
}

FeatureVector::FeatureVector(const FeatureVector &cpy)
{
    data = cpy.data;
    size = cpy.size;
}

FeatureVector& FeatureVector::operator=(const FeatureVector& rhs)
{
    if(this == &rhs)
        return *this;
    data = rhs.data;
    size = rhs.size;
    return *this;
}

FeatureVector& FeatureVector::operator=(const std::vector<double> rhs)
{
    data = rhs;
    size = rhs.size();
    return *this;
}

void FeatureVector::normalize()
{
    double mag = 0;
    for(size_t i = 0; i < size; ++i)
        mag += data[i]*data[i];
    mag = sqrt(mag);
    for(size_t i = 0; i < size; ++i)
        data[i] /= mag;
}

double FeatureVector::distance(const FeatureVector &fv, int type)
{
    assert(size == fv.size);
    double distance = 0;
    if(type == DISTANCE_TYPE_EUCLIDEAN)
    {
        for(size_t i = 0; i < size; ++i)
            distance += (data[i]-fv.data[i])*(data[i]-fv.data[i]);
        distance = sqrt(distance);
    }
    return distance;
}

////////////////////////////////////////////////////////////////////////
//FeatureHistogram
////////////////////////////////////////////////////////////////////////

FeatureHistogram::FeatureHistogram()
{
    size = 0;
    label = DEFAULT_LABEL_VALUE;
}

FeatureHistogram::~FeatureHistogram()
{

}

FeatureHistogram::FeatureHistogram(const FeatureHistogram &cpy)
{
    data = cpy.data;
    size = cpy.size;
    label = cpy.label;
}

////////////////////////////////////////////////////////////////////////
//FeatureCodex
////////////////////////////////////////////////////////////////////////

FeatureCodex::FeatureCodex()
{
    size = 0;
    length = 0;
}

FeatureCodex::FeatureCodex(const FeatureCodex &cpy)
{
    size = cpy.size;
    length = cpy.length;
    features = cpy.features;
    centroid = cpy.centroid;

    std::vector<cv::Mat> flannMat(1);
    flannMat[0].create(size, length, CV_32FC1);
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < length; ++j)
        {
            flannMat[0].at<float>(i,j) = features[i][j];
        }
    }

    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

}

FeatureCodex::~FeatureCodex()
{
    matcher.clear();
    size = 0;
    length = 0;
}

FeatureCodex& FeatureCodex::operator=(const FeatureCodex& rhs)
{
    if(this == &rhs)
        return *this;
    size = rhs.size;
    length = rhs.length;
    features = rhs.features;
    centroid = rhs.centroid;

    std::vector<cv::Mat> flannMat(1);
    flannMat[0].create(size, length, CV_32FC1);
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < length; ++j)
        {
            flannMat[0].at<float>(i,j) = features[i][j];
        }
    }

    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

    return *this;
}

void FeatureCodex::clear()
{
    features.clear();
    centroid.clear();
    size = 0;
    length = 0;
}

int FeatureCodex::match(const FeatureVector &f, int method)
{
    assert(length != f.size);
    if(method == MATCHING_FLANN)
        return FLANNMatch(f);
    else if(method == MATCHING_EUCLIDEAN)
        return EuclideanMatch(f);
    else return FLANNMatch(f);//put that in for now TODO: Fix this
}

int FeatureCodex::match(const std::vector<double> &f, int method)
{
    return match(FeatureVector(f), method);
}

void FeatureCodex::save(const string &filename)
{
    std::ofstream fout;
    fout.open(filename.c_str());
    fout << size << " " << length << endl;
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < length; ++j)
        {
            fout << features[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

bool FeatureCodex::load(const string &filename)
{
    std::ifstream fin;
    fin.open(filename.c_str());
    if(!fin.good())
        return false;

    fin >> size;
    fin >> length;

    vector<cv::Mat> flannMat(1);
    flannMat[0].create(size, length, CV_32FC1);

    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < length; ++j)
        {
            fin >> (features[i])[j];
            flannMat[0].at<float>(i,j) = features[i][j];
        }
    }
    fin.close();

    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

    return true;
}

void FeatureCodex::cluster(const std::vector<FeatureVector> &inputFeatures,
             const parameters::ClusteringParameters &params,
             int method)
{
    if(method == CLUSTERING_FLANN)
        FLANNCluster(inputFeatures, params);
}

void FeatureCodex::FLANNCluster(const std::vector<FeatureVector> &inputFeatures,
                          const parameters::ClusteringParameters &params)
{
    assert(inputFeatures.size() > 0);

    features.clear();
    centroid.clear();

    length = inputFeatures[0].size;
    size_t i, j;
    for(i = 1; i < inputFeatures.size(); ++i)
        assert(inputFeatures[i].size == length);

    assert(params.branching > 0 &&
           params.cbIndex > 0 &&
           params.numClusters > 0 &&
           params.numPass > 0);

    cv::Mat featureMat(inputFeatures.size(),length, CV_32F);
    cv::Mat clusterCenters(params.numClusters, length, CV_32F);
    cvflann::KMeansIndexParams paramsFLANN(params.branching, params.numPass, params.FLANNmethod, params.cbIndex);

    for(i = 0; i < inputFeatures.size(); ++i)
        for(j = 0; j < length; ++j)
            featureMat.at<float>(i,j) = inputFeatures[i][j];

    size = cv::flann::hierarchicalClustering<float,float>(featureMat, clusterCenters, paramsFLANN);

    vector<cv:: Mat> flannMat(1);
    flannMat[0] = clusterCenters;
    matcher.clear();
    matcher.add(flannMat);
    matcher.train();

    features.resize(size);
    for(i = 0; i < size; ++i)
    {
        features[i].resize(length);
        for(j = 0; j < length; ++j)
            features[i][j] = clusterCenters.at<float>(i,j);
    }
}


int FeatureCodex::FLANNMatch(const FeatureVector &f)
{
    vector<cv::DMatch> indexes;
    cv::Mat query(1, length, CV_32FC1);
    for(size_t i = 0; i < length; ++i)
        query.at<float>(0,i) = f[i];

    matcher.match(query, indexes);

    return indexes[0].trainIdx;
}

int FeatureCodex::EuclideanMatch(const FeatureVector &f)
{
    double dist;
    double minDist = DBL_MAX;
    int minIndex = 0;
    for(size_t i = 0; i < features.size(); ++i)
    {
        dist = features[i].distance(f);
        if(dist < minDist)
        {
            minDist = dist;
            minIndex = i;
        }
    }
    return minIndex;
}

}
