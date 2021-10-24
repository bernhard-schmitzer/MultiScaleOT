#ifndef PybindInterface_Common_H_
#define PybindInterface_Common_H_

#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Common/Verbose.h>

#include<pybind11/include/pybind11/pybind11.h>
#include<pybind11/include/pybind11/numpy.h>

namespace py = pybind11;


using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// basic data structure conversions

// create T<X>Matrix wrapper for numpy array
template<typename V>
TVMatrix<V>* getMatrixFromNumpyArray(py::array_t<V> &mu);


// short cut to get data pointer from numpy array
template<typename V>
V* getDataPointerFromNumpyArray(py::array_t<V> &mu);



// create new numpy arrays from raw arrays
template<class T>
py::array_t<T> getArrayFromRaw(const T * const data, int n);

// create new numpy arrays from raw arrays: multidim
template<class T>
py::array_t<T> getArrayFromRaw(const T * const data, std::vector<int> dimensions);

// extract python representation of data from TSparseCSRContainer
py::tuple getSparseCSRDataTuple(const TSparseCSRContainer &csrData);

// extract python representation of data from TSparsePosContainer
py::tuple getSparsePosDataTuple(const TSparsePosContainer &posData);

// extract python representation of data from TParticleContainer
py::tuple getParticleDataTuple(const TParticleContainer &particleData);


// create TSparseCSRContainer from raw data
TSparseCSRContainer getSparseCSRContainerFromData(
        const TDoubleMatrix * const data,
        const TInteger32Matrix * const indices, const TInteger32Matrix * const indptr,
        const int xres, const int yres);

// create TSparseCSRContainer from python data
TSparseCSRContainer getSparseCSRContainerFromData(
        py::array_t<double> &data,
        py::array_t<int> &indices, py::array_t<int> &indptr,
        const int xres, const int yres);


// create TSparsePosContainer from raw data
TSparsePosContainer getSparsePosContainerFromData(
        const TDoubleMatrix * const mass,
        const TInteger32Matrix * const posX, const TInteger32Matrix * const posY,
        const int xres, const int yres);

// create TSparsePosContainer from python data
TSparsePosContainer getSparsePosContainerFromData(
        py::array_t<double> &mass,
        py::array_t<int> &posX, py::array_t<int> &posY,
        const int xres, const int yres);

// create TParticleContainer from raw data
TParticleContainer getParticleContainerFromData(
        const TDoubleMatrix * const mass,
        const TDoubleMatrix * const pos);

// create TParticleContainer from python data
TParticleContainer getParticleContainerFromData(
        py::array_t<double> &mass,
        py::array_t<double> &pos);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract nChildren, children C array from python list
void extractListData(py::list pyData, std::vector<int> &counts, std::vector<int*> &data);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// testing argument and data validity helper functions

void test_nLayer(const double nLayer, const double nLayers);
// test if array is in C order, which is assumed by all of the underlying code
void test_Corder(const py::buffer_info &buffer);
void test_nMarginal(const int nMarginal, const int nMarginals);


#endif
