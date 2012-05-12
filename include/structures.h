#ifndef BOF_STRUCTURES_H
#define BOF_STRUCTURES_H

#include <vector>
#include <string>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "parameters.h"

namespace bof
{

const int DEFAULT_LABEL_VALUE = -1;

enum{DISTANCE_TYPE_EUCLIDEAN};
enum{CLUSTERING_FLANN, CLUSTERING_KMEANS};
enum{MATCHING_FLANN, MATCHING_EUCLIDEAN};

////////////////////////////////////////////////////////////////////////
//FeatureVector
////////////////////////////////////////////////////////////////////////

class FeatureVector
{
    public:
        FeatureVector();
        ~FeatureVector();
        FeatureVector(int _size)
        {
            resize(_size);
        }
        FeatureVector(const std::vector<double> &_data)
        {
            set(_data);
        }
        FeatureVector(const FeatureVector &cpy);

        void set(const std::vector<double> &_data)
        {
            data = _data;
            size = data.size();
        }
        void get(std::vector<double> &_data)
        {
            _data = data;
        }
        size_t getSize()
        {
            return size;
        }
        bool empty()
        {
            return !size;
        }
        void resize(size_t _size)
        {
            size = _size;
            data.resize(size);
        }
        void clear()
        {
            data.clear();
        }
        void zero()
        {
            for(size_t i = 0; i < size; ++i)
                data[i] = 0;
        }

        double& operator[](int idx)
        {
            assert(idx < (int)size && idx >= 0);
            return this->data[idx];
        }
        double operator[](int idx) const
        {
            assert(idx < (int)size && idx >= 0);
            return this->data[idx];
        }

        FeatureVector& operator=(const FeatureVector& rhs);
        FeatureVector& operator=(const std::vector<double> rhs);
        void normalize();
        double distance(const FeatureVector &fv, int type=DISTANCE_TYPE_EUCLIDEAN);

        //TODO this should be protected
        std::vector<double> data;
        size_t size;
};

////////////////////////////////////////////////////////////////////////
//FeatureHistogram
////////////////////////////////////////////////////////////////////////

class FeatureHistogram : public FeatureVector
{
    public:
        FeatureHistogram();
        ~FeatureHistogram();
        FeatureHistogram(int _size, int _label=DEFAULT_LABEL_VALUE)
        {
            resize(_size);
            label = _label;
        }
        FeatureHistogram(const std::vector<double> &_data, int _label=DEFAULT_LABEL_VALUE)
        {
            set(_data);
            label = _label;
        }
        FeatureHistogram(const FeatureHistogram &cpy);

        void setLabel(int _label)
        {
            label = _label;
        }
        int getLabel()
        {
            return label;
        }
        void addAt(int index, double value=1.0)
        {
            data[index] += value;
        }

        int label;
};

////////////////////////////////////////////////////////////////////////
//FeatureCodex
////////////////////////////////////////////////////////////////////////

class FeatureCodex
{
    public:
        FeatureCodex();
        FeatureCodex(const FeatureCodex &cpy);
        ~FeatureCodex();

        FeatureCodex& operator=(const FeatureCodex& rhs);
        FeatureVector& operator[](int idx)
        {
            assert(idx >= 0 && idx < (int)size);
            return features[idx];
        }

        void clear();
        size_t getSize()
        {
            return size;
        }

        void save(const string &filename);
        bool load(const string &filename);

        int match(const FeatureVector &f, int method=MATCHING_FLANN);
        int match(const std::vector<double> &f, int method=MATCHING_FLANN);
        void cluster(const std::vector<FeatureVector> &inputFeatures,
                     const parameters::ClusteringParameters &params,
                     int method=CLUSTERING_FLANN);


    private:
        std::vector<FeatureVector> features;
        FeatureVector centroid;
        cv::FlannBasedMatcher matcher;
        size_t size, length;

        void FLANNCluster(const std::vector<FeatureVector> &inputFeatures,
                          const parameters::ClusteringParameters &params);

        int FLANNMatch(const FeatureVector &f);
        int EuclideanMatch(const FeatureVector &f);

};

}//namespace bof

#endif
