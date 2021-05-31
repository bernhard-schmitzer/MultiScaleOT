#include"MultiScaleOT_pybind11_interface_common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// basic data structure conversions

// create T<X>Matrix wrapper for numpy array
template<typename V>
TVMatrix<V>* getMatrixFromNumpyArray(py::array_t<V> &mu) {
    TVMatrix<V> *result=new TVMatrix<V>();

    py::buffer_info buffer = mu.request();
    
    result->data=(V*) buffer.ptr;
    result->depth=buffer.ndim;
    result->dimensions=(int*) malloc(sizeof(int)*result->depth);
    for(int i=0;i<result->depth;i++) {
        result->dimensions[i]=(int) buffer.shape[i];
    }
    result->ownDimensions=true;
    result->ownData=false;
    return result;
}

template TVMatrix<double>* getMatrixFromNumpyArray<double>(py::array_t<double> &mu);
template TVMatrix<int>* getMatrixFromNumpyArray<int>(py::array_t<int> &mu);


// short cut to get data pointer from numpy array
template<typename V>
V* getDataPointerFromNumpyArray(py::array_t<V> &mu) {
    py::buffer_info buffer=mu.request();
    return (V*) buffer.ptr;
}

template double* getDataPointerFromNumpyArray<double>(py::array_t<double> &mu);
template int* getDataPointerFromNumpyArray<int>(py::array_t<int> &mu);


// create new numpy arrays from raw arrays
template<class T>
py::array_t<T> getArrayFromRaw(const T * const data, int n) {
    py::array_t<T> result = py::array_t<T>(n);
    std::memcpy(result.request().ptr,data,sizeof(T)*n);
    return result;
}

template py::array_t<double> getArrayFromRaw<double>(const double * const data, int n);
template py::array_t<int> getArrayFromRaw<int>(const int * const data, int n);

// create new numpy arrays from raw arrays: multidim
template<class T>
py::array_t<T> getArrayFromRaw(const T * const data, std::vector<int> dimensions) {
    int size=1;
    for(size_t i=0;i<dimensions.size();i++) {
        size*=dimensions[i];
    }
    py::array_t<T> result = py::array_t<T>(dimensions);
    std::memcpy(result.request().ptr,data,sizeof(T)*size);
    return result;
}

template py::array_t<double> getArrayFromRaw<double>(const double * const data, std::vector<int> dimensions);
template py::array_t<int> getArrayFromRaw<int>(const int * const data, std::vector<int> dimensions);


// extract python representation of data from TSparseCSRContainer
py::tuple getSparseCSRDataTuple(const TSparseCSRContainer &csrData) {

    py::array_t<double> resultData = py::array_t<double>(csrData.nonZeros);
    py::array_t<int> resultPos = py::array_t<int>(csrData.nonZeros);
    py::array_t<int> resultIndptr = py::array_t<int>(csrData.xres+1);
    std::memcpy(resultData.request().ptr,csrData.data.data(),sizeof(double)*csrData.nonZeros);
    std::memcpy(resultPos.request().ptr,csrData.indices.data(),sizeof(int)*csrData.nonZeros);
    std::memcpy(resultIndptr.request().ptr,csrData.indptr.data(),sizeof(int)*(csrData.xres+1));

    return py::make_tuple(resultData,resultPos,resultIndptr);
}

// extract python representation of data from TSparsePosContainer
py::tuple getSparsePosDataTuple(const TSparsePosContainer &posData) {

    py::array_t<double> resultMass = py::array_t<double>(posData.nParticles);
    py::array_t<int> resultPosStart = py::array_t<int>(posData.nParticles);
    py::array_t<int> resultPosEnd = py::array_t<int>(posData.nParticles);
    std::memcpy(resultMass.request().ptr,posData.mass.data(),sizeof(double)*posData.nParticles);
    std::memcpy(resultPosStart.request().ptr,posData.posStart.data(),sizeof(int)*posData.nParticles);
    std::memcpy(resultPosEnd.request().ptr,posData.posEnd.data(),sizeof(int)*posData.nParticles);

    return py::make_tuple(resultMass,resultPosStart,resultPosEnd);
}

// extract python representation of data from TParticleContainer
py::tuple getParticleDataTuple(const TParticleContainer &particleData) {

    py::array_t<double> resultMass = getArrayFromRaw<double>(particleData.mass.data(), particleData.nParticles);
    std::vector<int> dimensions={particleData.nParticles,particleData.dim};
    py::array_t<double> resultPos = getArrayFromRaw<double>(particleData.pos.data(), dimensions);

    return py::make_tuple(resultMass,resultPos);
}

// create TSparsePosContainer from raw data
TSparsePosContainer getSparsePosContainerFromData(
        const TDoubleMatrix * const mass,
        const TInteger32Matrix * const posX, const TInteger32Matrix * const posY,
        const int xres, const int yres) {
    TSparsePosContainer result;
    result.posStart.resize(posX->dimensions[0]);
    result.posEnd.resize(posY->dimensions[0]);
    result.mass.resize(mass->dimensions[0]);
    result.xres=xres;
    result.yres=yres;
    result.nParticles=posX->dimensions[0];
    std::memcpy(result.posStart.data(),posX->data,sizeof(int)*result.nParticles);
    std::memcpy(result.posEnd.data(),posY->data,sizeof(int)*result.nParticles);
    std::memcpy(result.mass.data(),mass->data,sizeof(double)*result.nParticles);
    
    return result;
}


// create TSparsePosContainer from python data
TSparsePosContainer getSparsePosContainerFromData(
        py::array_t<double> &mass,
        py::array_t<int> &posX, py::array_t<int> &posY,
        const int xres, const int yres) {

    py::buffer_info bufferPosX = posX.request();
    py::buffer_info bufferPosY = posY.request();
    py::buffer_info bufferMass = mass.request();


    TSparsePosContainer result;
    result.posStart.resize(bufferPosX.shape[0]);
    result.posEnd.resize(bufferPosY.shape[0]);
    result.mass.resize(bufferMass.shape[0]);
    result.xres=xres;
    result.yres=yres;
    result.nParticles=bufferPosX.shape[0];
    std::memcpy(result.posStart.data(),bufferPosX.ptr,sizeof(int)*result.nParticles);
    std::memcpy(result.posEnd.data(),bufferPosY.ptr,sizeof(int)*result.nParticles);
    std::memcpy(result.mass.data(),bufferMass.ptr,sizeof(double)*result.nParticles);
    
    return result;
}

// create TParticleContainer from raw data
TParticleContainer getParticleContainerFromData(
        const TDoubleMatrix * const mass,
        const TDoubleMatrix * const pos) {
    TParticleContainer result;
    result.nParticles=mass->dimensions[0];
    result.dim=pos->dimensions[1];
    result.mass.resize(result.nParticles);
    result.pos.resize(result.nParticles*result.dim);
    std::memcpy(result.mass.data(),mass->data,sizeof(double)*result.nParticles);
    std::memcpy(result.pos.data(),pos->data,sizeof(double)*result.nParticles*result.dim);
    
    return result;
}

// create TParticleContainer from python data
TParticleContainer getParticleContainerFromData(
        py::array_t<double> &mass,
        py::array_t<double> &pos) {

    py::buffer_info bufferMass = mass.request();
    py::buffer_info bufferPos = pos.request();

    TParticleContainer result;
    result.nParticles=bufferMass.shape[0];
    result.dim=bufferPos.shape[1];
    result.mass.resize(result.nParticles);
    result.pos.resize(result.nParticles*result.dim);
    std::memcpy(result.mass.data(),bufferMass.ptr,sizeof(double)*result.nParticles);
    std::memcpy(result.pos.data(),bufferPos.ptr,sizeof(double)*result.nParticles*result.dim);
    
    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// testing argument and data validity helper functions

void test_nLayer(const double nLayer, const double nLayers) {
    if (nLayer>=nLayers) throw std::out_of_range("nLayer too large");
    if (nLayer<0) throw std::out_of_range("nLayer too small");
}

// test if array is in C order, which is assumed by all of the underlying code
void test_Corder(const py::buffer_info &buffer) {
    int expected_size=buffer.itemsize;
    for(int i=buffer.ndim-1;i>=0;i--) {
        if(expected_size!=buffer.strides[i]) throw std::invalid_argument("Array data not in C order.");
        expected_size*=buffer.shape[i];
    }
}


void test_nMarginal(const int nMarginal, const int nMarginals) {
    if (nMarginal>=nMarginals) throw std::out_of_range("Index nMarginal of marginal too large");
    if (nMarginal<0) throw std::out_of_range("Index nMarginal of marginal too small");
}


