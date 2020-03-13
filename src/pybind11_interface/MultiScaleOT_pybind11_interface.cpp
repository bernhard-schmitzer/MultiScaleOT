#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Common/Verbose.h>

#ifdef USE_SINKHORN
#include<Sinkhorn.h>
#endif


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
// advanced kernel manipulation
#ifdef USE_SINKHORN
py::tuple findKernelLine(TSinkhornKernelGenerator *kernelGenerator, int layerFinest, int a, int mode, double slack) {
    
    std::vector<TSinkhornKernelGenerator::TCandidate> kernelLine=kernelGenerator->findKernelLine(layerFinest,a,mode,slack);
    py::array_t<int> resultPos(kernelLine.size());
    py::array_t<double> resultVal(kernelLine.size());
    
    int *resultPosPtr=getDataPointerFromNumpyArray(resultPos);
    double *resultValPtr=getDataPointerFromNumpyArray(resultVal);
    
    for(size_t i=0;i<kernelLine.size();i++) {
        resultPosPtr[i]=kernelLine[i].z;
        resultValPtr[i]=kernelLine[i].v;
    }
    
    return py::make_tuple(resultPos,resultVal);
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// post processing and interpolation
TParticleContainer PyInterpolateEuclidean(
        TSparsePosContainer &couplingData, double *posX, double *posY, int dim, double t) {
 
    TGeometry_Euclidean geometry;
    
    TParticleContainer result=ModelOT_Interpolate<TGeometry_Euclidean>(couplingData,
            posX, posY,
            dim, t, geometry);

    return result;

}


TParticleContainer PyInterpolateEuclideanWFR(
        TSparsePosContainer &couplingData,
        double *muXEff, double *muYEff,
        double *muX, double *muY,
        double *posX, double *posY, int dim, double t, double WFScale) {
 
    TGeometry_Euclidean geometry;
        
    TParticleContainer result=ModelWFR_Interpolate<TGeometry_Euclidean>(
            couplingData,
            muXEff, muYEff, muX, muY,
            posX, posY,
            dim, t, WFScale, geometry);

    return result;

}


void PyProjectInterpolation(TParticleContainer &particles, TDoubleMatrix *img) {
    if (particles.dim==1) {
        ProjectInterpolation<1>(particles.pos.data(), particles.mass.data(),
            img->data, particles.nParticles, img->dimensions);
    } else if (particles.dim==2) {
        ProjectInterpolation<2>(particles.pos.data(), particles.mass.data(),
            img->data, particles.nParticles, img->dimensions);
    }
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// module definition


PYBIND11_MODULE(MultiScaleOT, m) {
    m.doc() = "pybind11 interface to MultiScaleOT library"; // module docstring

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TDoubleMatrix>(m, "TDoubleMatrix")
        .def(py::init([](py::array_t<double> &mu) {
            return getMatrixFromNumpyArray<double>(mu);
        }))
        .def("getDepth",
        [](const TDoubleMatrix *a) {
            return a->depth;
        })
        .def("getDimensions",
        [](const TDoubleMatrix *a) {
            py::array_t<int> result(a->depth);
            std::memcpy((int*) result.request().ptr,a->dimensions,a->depth*sizeof(int));
            return result;
        });


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TInteger32Matrix>(m, "TInteger32Matrix")
        .def(py::init([](py::array_t<int> &mu) {
            return getMatrixFromNumpyArray<int>(mu);
        }))
        .def("getDepth",
        [](const TInteger32Matrix *a) {
            return a->depth;
        })
        .def("getDimensions",
        [](const TInteger32Matrix *a) {
            py::array_t<int> result(a->depth);
            std::memcpy((int*) result.request().ptr,a->dimensions,a->depth*sizeof(int));
            return result;
        });


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #ifdef USE_SINKHORN
    py::class_<TKernelMatrix>(m, "TKernelMatrix")
        .def("getCSRData",[](const TKernelMatrix &kernel) {
            return SinkhornKernelGetCSRData(kernel);
        },"Returns a TSparseCSRContainer object containing the non-zero kernel entries in sparse CSR format.")
        .def("getCSRDataTuple",[](const TKernelMatrix &kernel) {
            return getSparseCSRDataTuple(SinkhornKernelGetCSRData(kernel));
        },"Returns a tuple of numpy arrays (data,indices,indptr) of types (float64,int32,int32) containing the non-zero kernel entries in sparse CSR format.")
        .def("getPosData",[](const TKernelMatrix &kernel) {
            return SinkhornKernelGetPosData(kernel);
        },"Returns a TSparsePosContainer object containing the non-zero kernel entries in sparse POS format.")
        .def("getPosDataTuple",[](const TKernelMatrix &kernel) {
            return getSparsePosDataTuple(SinkhornKernelGetPosData(kernel));
        },"Returns a tuple of numpy arrays (posStart,posEnd,mass) of types (int32,int32,float64) containing the non-zero kernel entries in sparse POS format (row indices, column indices, mass values).");
    #endif // Sinkhorn
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSparseCSRContainer>(m,"TSparseCSRContainer")
        .def("getDataTuple",[](const TSparseCSRContainer &csrContainer) {
            return getSparseCSRDataTuple(csrContainer);
        });


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSparsePosContainer>(m,"TSparsePosContainer")
        .def(py::init([](py::array_t<double> &mass, py::array_t<int> &posX, py::array_t<int> &posY,
                const int xres, const int yres) {
            return getSparsePosContainerFromData(mass,posX,posY,xres,yres);
            }),
            py::arg("mass"),py::arg("posX"),py::arg("posY"),py::arg("xres"),py::arg("yres")
            )
        .def("getDataTuple",[](const TSparsePosContainer &posContainer) {
            return getSparsePosDataTuple(posContainer);
        });
 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TParticleContainer>(m,"TParticleContainer")
        .def(py::init([](py::array_t<double> &mass, py::array_t<double> &pos) {
            return getParticleContainerFromData(mass,pos);
            }),
            py::arg("mass"),py::arg("pos")
            )
        .def("getDataTuple",[](const TParticleContainer &particleContainer) {
            return getParticleDataTuple(particleContainer);
        });

    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    m.attr("childModeGrid")=THierarchyBuilder::CM_Grid;
    m.attr("childModeTree")=THierarchyBuilder::CM_Tree;
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TMultiScaleSetup>(m, "TMultiScaleSetup","Multiscale representation of a measure supported on a point cloud.")
        ///////////////////////////////////////////
        .def(py::init([](py::array_t<double> &pos, py::array_t<double> &mu, int depth, int childMode,
                    bool setup, bool setupDuals, bool setupRadii) {
                py::buffer_info posBuffer=pos.request();
                py::buffer_info muBuffer=mu.request();
                
                // data validity checks
                if(posBuffer.ndim!=2) throw std::invalid_argument("pos must be 2d array.");
                if(muBuffer.ndim!=1) throw std::invalid_argument("mu must be 1d array.");
                if(muBuffer.shape[0]!=posBuffer.shape[0]) throw std::invalid_argument("pos and mu have incompatible shapes.");
                test_Corder(posBuffer);
                test_Corder(muBuffer);
                
                TMultiScaleSetup *result= new TMultiScaleSetup((double*) posBuffer.ptr, (double*) muBuffer.ptr, posBuffer.shape[0], posBuffer.shape[1], depth, childMode,
                        setup, setupDuals, setupRadii);
                return result;
            }),
            py::arg("pos"), py::arg("mu"), py::arg("depth"),
            py::arg("childMode")=THierarchyBuilder::CM_Grid,
            py::arg("setup")=true, py::arg("setupDuals")=true, py::arg("setupRadii")=true,
            R"(
            Args:
                pos: 2d double array of point locations where the measure is supported
                mu: 1d double array of point masses
                depth: multiscale representation will have depth+1 layers. depth=0 means no multiscale representation, just the original point cloud.
                childMode: determines the order of children nodes in intermediate layers.

                    * 0 (=tree): adaptive 2^n tree where empty branches are discarded
                    * 1 (=grid): every layer is assumed to be a full Cartesian grid, nodes are numbered and stored in C order. Simpler if applicable, only works if all nodes are actually required.

                setup: whether constructor automatically runs basic setup() method
                setupDuals: whether constructor allocates storage for dual variables (can be done later with method setupDuals() )
                setupRadii: whether constructor computes radii of hierarchical partition cells (can be done later with setupRadii() ))"
            )
        ///////////////////////////////////////////
        .def("setup", &TMultiScaleSetup::Setup,"Basic setup after calling the constructor. Instance cannot be used without calling this first, but constructor by default calls it automatically. See also constructor.")
        .def("setupDuals", &TMultiScaleSetup::SetupDuals,"Allocate memory for dual variables (at all scales). See also constructor.")
        .def("setupRadii", &TMultiScaleSetup::SetupRadii,"Compute radii of hierarchical partition cells. See also constructor.")
        .def("getDepth", [](const TMultiScaleSetup * const a) { return a->depth; },"Returns the value of the depth parameter of the TMultiScaleSetup instance (which is <number of layers>-1).")
        .def("getNLayers", [](const TMultiScaleSetup * const a) { return a->nLayers; },
        "Returns the number of layers of the TMultiScaleSetup instance (which is depth+1).")
        .def("getNPoints", [](const TMultiScaleSetup *const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers);
            return a->HP->layers[nLayer]->nCells;
            },
            py::arg("nLayer"),
            "Get number of points at layer nLayer.")
        .def("getPoints", [](const TMultiScaleSetup * const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers);
            std::vector<int> dimensions={a->HP->layers[nLayer]->nCells,a->dim};
            return getArrayFromRaw<double>(a->posH[nLayer], dimensions);
            },
            py::arg("nLayer"),
            "Get points at layer nLayer. If nLayer==nLayers-1==depth it returns the original point cloud.")
        .def("getMeasure", [](const TMultiScaleSetup * const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers);
            return getArrayFromRaw<double>(a->muH[nLayer], a->HP->layers[nLayer]->nCells);
            },
            py::arg("nLayer"),
            "Get measure at layer nLayer. If nLayer==nLayers-1==depth it returns the original measure.")
        .def("getParents", [](const TMultiScaleSetup * const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers);
            return getArrayFromRaw<int>(a->HP->layers[nLayer]->parent, a->HP->layers[nLayer]->nCells);
            },
            py::arg("nLayer"),
            R"(
            Get parents of nodes at layer nLayer.
            
            Returns zeros at coarsest layer, nLayer=0.
            At finer layers returns array of int32 of size of current layer. Each entry gives id of parent node in parent layer. This id is location of the parent node in points or measure arrays.)")
        .def("getRadii", [](const TMultiScaleSetup * const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers-1);
            return getArrayFromRaw<double>(a->radii[nLayer], a->HP->layers[nLayer]->nCells);
            },
            py::arg("nLayer"),
            "Get radii at layer nLayer. Only valid for nLayer<=nLayers-2 since there are no radii on finest layer.")
        .def("getChildren", [](const TMultiScaleSetup * const a, const int nLayer, const int id) {
            test_nLayer(nLayer,a->nLayers-1);
            if(id<0) throw std::invalid_argument("id cannot be negative.");
            if(id>=a->HP->layers[nLayer]->nCells) throw std::invalid_argument("id larger than number of nodes at that layer.");
            return getArrayFromRaw<int>(a->HP->layers[nLayer]->children[id], a->HP->layers[nLayer]->nChildren[id]);
            },
            py::arg("nLayer"), py::arg("id"),
            "Get list of children of node id at layer nLayer.")
        .def("getDual", [](const TMultiScaleSetup * const a, const int nLayer) {
            test_nLayer(nLayer,a->nLayers);
            return getArrayFromRaw<double>(a->alphaH[nLayer], a->HP->layers[nLayer]->nCells);
            },
            py::arg("nLayer"),
            "Return dual variable values at layer nLayer.")
        .def("setChildModeGrid", [](TMultiScaleSetup *a) {
            a->HierarchyBuilderChildMode=THierarchyBuilder::CM_Grid;
            },"Sets childMode to 0 (=tree). Only meaningful before setup() is called, see constructor.")
        .def("setChildModeTree", [](TMultiScaleSetup *a) {
            a->HierarchyBuilderChildMode=THierarchyBuilder::CM_Tree;
            },"Sets childMode to 1 (=grid). Only meaningful before setup() is called, see constructor.")
        .def("setInterpolationMode", [](TMultiScaleSetup *a, int interpolation_mode) {
            a->HP->interpolation_mode=interpolation_mode;
            },py::arg("interpolation_mode"),"Experimental undocumented method.")
        .def("setDual", [](TMultiScaleSetup *a, py::array_t<double> &alpha, const int nLayer) {
            py::buffer_info alphaBuffer=alpha.request();            
            test_nLayer(nLayer,a->nLayers);
            if(alphaBuffer.ndim!=1) throw std::invalid_argument("alpha must be 1d array.");
            const int xres=a->HP->layers[nLayer]->nCells;
            if(xres!=alphaBuffer.shape[0]) throw std::invalid_argument("alpha size does not match layer size.");

            std::memcpy(a->alphaH[nLayer],alphaBuffer.ptr,sizeof(double)*xres);
            a->HP->signal_propagate_double(a->alphaH, 0, nLayer, THierarchicalPartition::MODE_MAX);
            },
            py::arg("alpha"), py::arg("nLayer"),
            R"(
            Set dual variable at layer nLayer to alpha and then propagates values to all coarser layers by maximization over child nodes.
            
            Args:
                alpha: 1d double array with new values for dual variable. Size needs to equal number of cells in that layer.
                nLayer: id of layer on which dual variable is to be set.)")
        .def("refineSignal", [](TMultiScaleSetup *a, py::array_t<double> &signal, const int lTop, const int mode) {
            if(lTop>a->HP->nLayers-1) throw std::invalid_argument("lTop too large.");

            const int xresCoarse=a->HP->layers[lTop]->nCells;
            py::buffer_info signalBuffer=signal.request();

            if(signalBuffer.ndim!=1) throw std::invalid_argument("signal must be 1d array.");
            if(xresCoarse!=signalBuffer.shape[0]) throw std::invalid_argument("signal size does not match coarse layer size.");

            // allocate memory for refined signal
            const int xresFine=a->HP->layers[lTop+1]->nCells;
            py::array_t<double> signalFine(xresFine);
            py::buffer_info signalFineBuffer=signalFine.request();

            a->HP->signal_refine_double((double*) signalBuffer.ptr, (double*) signalFineBuffer.ptr, lTop,mode);
            
            return signalFine;
            },
            py::arg("signal"), py::arg("lTop"), py::arg("mode")=THierarchicalPartition::INTERPOLATION_MODE_CONSTANT,
            R"(
            Refines a hierarchical signal from one layer to the subsequent finer layer.
            
            Args:
                signal: 1d double array with signal at coarse layer
                lTop: id of coarse layer
                mode: determines how signal is refined.

                    * 0: fine value equals parent value (default)
                    * 1: fine value is interpolated piecewise linearly (only works on Cartesian grids constructed with childMode=grid, experimental))

            Returns:
                1d double array containing refined signal at fine layer (one layer below coarse layer))");
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TMultiScaleSetupGrid,TMultiScaleSetup>(m, "TMultiScaleSetupGrid")
        .def(py::init([](py::array_t<double> &muGrid, int depth,
                    bool setup, bool setupDuals, bool setupRadii) {

                py::buffer_info muBuffer=muGrid.request();
                test_Corder(muBuffer);

                TDoubleMatrix *muGridMat=getMatrixFromNumpyArray<double>(muGrid);
                TMultiScaleSetupGrid *result= new TMultiScaleSetupGrid(muGridMat, depth,
                        setup, setupDuals, setupRadii);
                result->ownMuGrid=true; // this instance of TMultiScaleSetupGrid should take care of muGridMat memory upon deletion
                return result;
            }),
            py::arg("muGrid"), py::arg("depth"),
            py::arg("setup")=true, py::arg("setupDuals")=true, py::arg("setupRadii")=true
            );
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


    py::class_<TMultiScaleSetupBarycenterContainer>(m, "TMultiScaleSetupBarycenterContainer")
        .def(py::init([]() {return new TMultiScaleSetupBarycenterContainer();}))
        .def(py::init([](const int nMarginals) { return new TMultiScaleSetupBarycenterContainer(nMarginals); }),
            py::arg("nMarginals"))
        .def("setMarginal", [](TMultiScaleSetupBarycenterContainer &a, const int nMarginal, TMultiScaleSetup &multiScaleSetup, const double weight) {
            test_nMarginal(nMarginal, a.nMarginals);
            a.setMarginal(nMarginal, multiScaleSetup, weight);
            },
            py::arg("nMarginal"), py::arg("MultiScaleSetup"), py::arg("weight"),
            "Set nMarginal-th marginal in barycenter problem to MultiScaleSetup with weight-coefficient weight."
            )
        .def("setCenterMarginal", &TMultiScaleSetupBarycenterContainer::setCenterMarginal)
        .def("setupDuals", &TMultiScaleSetupBarycenterContainer::setupDuals)
        .def("setCostFunctionProvider", [](TMultiScaleSetupBarycenterContainer &a, const int nMarginal, THierarchicalCostFunctionProvider &costFunctionProvider) {
            test_nMarginal(nMarginal, a.nMarginals);
            a.setCostFunctionProvider(nMarginal, costFunctionProvider);
            },
            py::arg("nMarginal"), py::arg("costFunctionProvider"),
            "Set nMarginal-th cost function in barycenter problem to costFunctionProvider."
            )
        .def("getAlpha", [](const TMultiScaleSetupBarycenterContainer &a, const int nMarginal, const int nLayer) {
            test_nMarginal(nMarginal, a.nMarginals);
            test_nLayer(nLayer, a.HP[nMarginal]->nLayers);
            return getArrayFromRaw<double>(a.alphaH[nMarginal][nLayer], a.HP[nMarginal]->layers[nLayer]->nCells);
            }, py::arg("nMarginal"), py::arg("nLayer"))
        .def("getBeta", [](const TMultiScaleSetupBarycenterContainer &a, const int nMarginal, const int nLayer) {
            test_nMarginal(nMarginal, a.nMarginals);
            test_nLayer(nLayer, a.HPZ->nLayers);
            return getArrayFromRaw<double>(a.betaH[nMarginal][nLayer], a.HPZ->layers[nLayer]->nCells);
            }, py::arg("nMarginal"), py::arg("nLayer"))
        .def("setAlpha", [](TMultiScaleSetupBarycenterContainer &a, py::array_t<double> &alpha, const int nMarginal, const int nLayer) {
            test_nMarginal(nMarginal, a.nMarginals);
            test_nLayer(nLayer, a.HP[nMarginal]->nLayers);
            py::buffer_info alphaBuffer=alpha.request();            
            if(alphaBuffer.ndim!=1) throw std::invalid_argument("alpha must be 1d array.");
            const int xres=a.HP[nMarginal]->layers[nLayer]->nCells;
            if(xres!=alphaBuffer.shape[0]) throw std::invalid_argument("alpha size does not match layer size.");

            std::memcpy(a.alphaH[nMarginal][nLayer],alphaBuffer.ptr,sizeof(double)*xres);
            a.HP[nMarginal]->signal_propagate_double(a.alphaH[nMarginal], 0, nLayer, THierarchicalPartition::MODE_MAX);
            },
            py::arg("alpha"), py::arg("nMarginal"), py::arg("nLayer"),
            R"(
            Set dual variable alpha of marginal nMarginal at layer nLayer to alpha and then propagates values to all coarser layers by maximization over child nodes.
            
            Args:
                alpha: 1d double array with new values for dual variable. Size needs to equal number of cells in that layer.
                nMarginal: id of marginal on which dual variable is to be set.
                nLayer: id of layer on which dual variable is to be set.)")
        .def("setBeta", [](TMultiScaleSetupBarycenterContainer &a, py::array_t<double> &beta, const int nMarginal, const int nLayer) {
            test_nMarginal(nMarginal, a.nMarginals);
            test_nLayer(nLayer, a.HPZ->nLayers);
            py::buffer_info betaBuffer=beta.request();            
            if(betaBuffer.ndim!=1) throw std::invalid_argument("beta must be 1d array.");
            const int xres=a.HPZ->layers[nLayer]->nCells;
            if(xres!=betaBuffer.shape[0]) throw std::invalid_argument("beta size does not match layer size.");

            std::memcpy(a.betaH[nMarginal][nLayer],betaBuffer.ptr,sizeof(double)*xres);
            a.HPZ->signal_propagate_double(a.betaH[nMarginal], 0, nLayer, THierarchicalPartition::MODE_MAX);
            },
            py::arg("beta"), py::arg("nMarginal"), py::arg("nLayer"),
            R"(
            Set dual variable beta of marginal nMarginal at layer nLayer to beta and then propagates values to all coarser layers by maximization over child nodes.
            
            Args:
                beta: 1d double array with new values for dual variable. Size needs to equal number of cells in that layer.
                nMarginal: id of marginal on which dual variable is to be set.
                nLayer: id of layer on which dual variable is to be set.)");


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    #ifdef USE_SINKHORN
    py::class_<TSinkhornSolverBase::TSinkhornSolverParameters>(m, "TSinkhornSolverParameters","Bundle all smaller parameters of the Sinkhorn solver objects.")
        .def(py::init<int,int,int,double,double,double,bool,bool>(),
            py::arg("maxIterations")=TSinkhornSolverBase::SinkhornSolverParametersDefault.maxIterations,
            py::arg("innerIterations")=TSinkhornSolverBase::SinkhornSolverParametersDefault.innerIterations,
            py::arg("maxAbsorptionLoops")=TSinkhornSolverBase::SinkhornSolverParametersDefault.maxAbsorptionLoops,
            py::arg("absorption_scalingBound")=TSinkhornSolverBase::SinkhornSolverParametersDefault.absorption_scalingBound,
            py::arg("absorption_scalingLowerBound")=TSinkhornSolverBase::SinkhornSolverParametersDefault.absorption_scalingLowerBound,
            py::arg("truncation_thresh")=TSinkhornSolverBase::SinkhornSolverParametersDefault.truncation_thresh,
            py::arg("refineKernel")=TSinkhornSolverBase::SinkhornSolverParametersDefault.refineKernel,
            py::arg("generateFinalKernel")=TSinkhornSolverBase::SinkhornSolverParametersDefault.generateFinalKernel
            )
        .def_readwrite("maxIterations", &TSinkhornSolverBase::TSinkhornSolverParameters::maxIterations)
        .def_readwrite("innerIterations", &TSinkhornSolverBase::TSinkhornSolverParameters::innerIterations)
        .def_readwrite("maxAbsorptionLoops", &TSinkhornSolverBase::TSinkhornSolverParameters::maxAbsorptionLoops)
        .def_readwrite("absorption_scalingBound", &TSinkhornSolverBase::TSinkhornSolverParameters::absorption_scalingBound)
        .def_readwrite("absorption_scalingLowerBound", &TSinkhornSolverBase::TSinkhornSolverParameters::absorption_scalingLowerBound)
        .def_readwrite("truncation_thresh", &TSinkhornSolverBase::TSinkhornSolverParameters::truncation_thresh)
        .def_readwrite("refineKernel", &TSinkhornSolverBase::TSinkhornSolverParameters::refineKernel)
        .def_readwrite("generateFinalKernel", &TSinkhornSolverBase::TSinkhornSolverParameters::generateFinalKernel);
    #endif // Sinkhorn

        
    py::class_<TEpsScalingHandler>(m, "TEpsScalingHandler","Class for setting up and managing a schedule for epsilon scaling.")
        .def(py::init([]() {
            TEpsScalingHandler *result=new TEpsScalingHandler();
            return result;
            })
            )
        .def("setupGeometricMultiLayerA", &TEpsScalingHandler::setupGeometricMultiLayerA,
            py::arg("nLayers"), py::arg("epsStart"), py::arg("epsTarget"), py::arg("epsSteps"), py::arg("boxScale"), py::arg("layerExponent"), py::arg("layerCoarsest"), py::arg("overlap")
            )
        .def("setupGeometricMultiLayerB", &TEpsScalingHandler::setupGeometricMultiLayerB,
            py::arg("nLayers"), py::arg("epsBase"), py::arg("layerFactor"), py::arg("layerSteps"), py::arg("stepsFinal")
            )
        .def("setupGeometricSingleLayer", &TEpsScalingHandler::setupGeometricSingleLayer,
            py::arg("nLayers"), py::arg("epsStart"), py::arg("epsTarget"), py::arg("epsSteps")
            )
        .def("setupExplicit", [](TEpsScalingHandler *a, py::list pyEpsLists) {
            int nLayers=pyEpsLists.size();
            double **epsLists=(double**) malloc(sizeof(double*)*nLayers);
            int *nEpsLists=(int*) malloc(sizeof(int)*nLayers);
            for(int i=0;i<nLayers;i++) {
                py::buffer_info buffer=pyEpsLists[i].cast<py::array_t<double>>().request();
                nEpsLists[i]=buffer.shape[0];
                epsLists[i]=(double*) buffer.ptr;
            }
            a->setupExplicit(nLayers, epsLists, nEpsLists);
            free(epsLists);
            free(nEpsLists);
            },
            py::arg("epsLists")
            )
        .def("get",[](const TEpsScalingHandler &a) {
            py::list result;
            for(int i=0;i<a.nLayers;i++) {
                result.append(getArrayFromRaw<double>(a.epsLists[i], a.nEpsLists[i]));
            }
            return result;
            });
           

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<THierarchicalCostFunctionProvider>(m, "THierarchicalCostFunctionProvider","Abstract base class of cost function object. Does not expose a constructor.")
        .def("getCost", &THierarchicalCostFunctionProvider::getCost, py::arg("layer"), py::arg("xId"), py::arg("yId"),"Computes lower bound on cost between points with numbers <xId> and <yId> at layer <layer>.")
        .def("getCostEff", &THierarchicalCostFunctionProvider::getCostEff, py::arg("layer"), py::arg("xId"), py::arg("yId"),"Computes lower bound on effective cost (i.e. cost reduced by dual variables) between points with numbers <xId> and <yId> at layer <layer>.")
        .def("getCostAsym", &THierarchicalCostFunctionProvider::getCostAsym, py::arg("xLayer"), py::arg("xId"), py::arg("yLayer"), py::arg("yId"),"Computes lower bound on cost between points with numbers <xId> at layer <xLayer> and <yId> at layer <yLayer>.")
        .def("getCostEffAsym", &THierarchicalCostFunctionProvider::getCostEffAsym, py::arg("xLayer"), py::arg("xId"), py::arg("yLayer"), py::arg("yId"),"Computes lower bound on effective cost (i.e. cost reduced by dual variables) between points with numbers <xId> at layer <xLayer> and <yId> at layer <yLayer>.")
        .def("setLayerBottom", &THierarchicalCostFunctionProvider::setLayerBottom, py::arg("layerBottom"),"Sets number of layer which should be considered as finest level.");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<THierarchicalCostFunctionProvider_SquaredEuclidean, THierarchicalCostFunctionProvider>
            (m, "THierarchicalCostFunctionProvider_SquaredEuclidean","Represents the squared Euclidean distance cost function.")
        //////////////////////////////////////////////////////
        .def(py::init([](
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                bool WFmode, double WFlenscale) {
            return new THierarchicalCostFunctionProvider_SquaredEuclidean(
                     MultiScaleSetupX->posH, MultiScaleSetupY->posH,
                    MultiScaleSetupX->radii, MultiScaleSetupY->radii,
                    MultiScaleSetupX->dim, MultiScaleSetupX->nLayers-1,
                    true,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    1.,
                    WFmode, WFlenscale
                    );
        }),
            py::arg("MultiScaleSetupX"),py::arg("MultiScaleSetupY"),py::arg("WFmode")=false,py::arg("WFlenscale")=1.,
            R"(
            Args:
                MultiScaleSetupX: TMultiScaleSetup instance describing first marginal
                MultiScaleSetupY: TMultiScaleSetup instance describing second marginal
                WFmode: whether Hellinger--Kantorovich mode should be activated
                WFlenscale: trade-off weight between transport and mass change. Maximal transport distance is WFlenscale*pi/2.
            )"
        )
        //////////////////////////////////////////////////////
        .def_readwrite("WFmode", &THierarchicalCostFunctionProvider_SquaredEuclidean::WFmode)
        .def_readwrite("layerBottom", &THierarchicalCostFunctionProvider_SquaredEuclidean::layerBottom)
        .def("setScale",&THierarchicalCostFunctionProvider_SquaredEuclidean::setScale)
        .def("setWFScale",&THierarchicalCostFunctionProvider_SquaredEuclidean::setWFScale);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    #ifdef USE_SINKHORN
    py::class_<TSinkhornSolverBase>(m,"TSinkhornSolverBase",
            R"("
            This is an abstract base class for the specialized SinkhornSolver classes.
            It does not expose a constructor to the user but its methods are inherited by all specialized Sinkhorn solver classes.)"
            )
        .def_readwrite("errorGoal", &TSinkhornSolverBase::errorGoal,"Double. Target accuracy for stopping the Sinkhorn algorithm.")
        .def_readwrite("kernelValid", &TSinkhornSolverBase::kernelValid,
            R"(
            Boolean. Is the currently stored kernel object valid (e.g. was epsilon changed or have absorbed dual variables changed since last computation).
            Experimental. Only experts should modify this.)"
            )
        .def("refineDuals", &TSinkhornSolverBase::refineDuals,py::arg("newLayer"),"Refines duals from coarse layer <newLayer-1> to <newLayer>. Experimental.")
        .def("solve", &TSinkhornSolverBase::solve,"Run the full main algorithm to solve the problem.")
        .def("solveSingle", &TSinkhornSolverBase::solveSingle,"Solve for current epsilon at current layer. Experimental.")
        .def("solveLayer", &TSinkhornSolverBase::solveLayer,"Solve over epsilon list at current layer. Experimental.")
        .def("changeLayer", &TSinkhornSolverBase::changeLayer,py::arg("newLayer"),"Set current layer. Experimental.")
        .def("changeEps", &TSinkhornSolverBase::changeEps,py::arg("newEps"),"Set current epsilon. Experimental.")
        .def("initialize", &TSinkhornSolverBase::initialize,"Must be called after constructor before any other methods.")
        .def("iterate", &TSinkhornSolverBase::iterate,
                py::arg("n"),"Perform n iterations.")
        .def("generateKernel", &TSinkhornSolverBase::generateKernel,"Recompute truncated kernel matrix based on current dual variables.")
        .def("absorb", &TSinkhornSolverBase::absorb, "Absorb current values of scaling factors into dual variables and reset scaling factors to 1. After calling this, one should usually set kernelValid to false.")
        .def("checkAbsorb", &TSinkhornSolverBase::checkAbsorb,
        py::arg("maxValue"),
        R"(
        Tests whether values of the scaling factors exceed maxValue. If so, an absorption and subsequent kernel computation are recommended.

        Args:
            maxValue: threshold for scaling factors

        Returns:
              * 1 if maxValue is exceeded
              * 0 otherwise
              * other values indicate an error)"
        )
        .def("getEps", [](const TSinkhornSolverBase *a) {
            return a->eps;
            },"Return current value of epsilon.")
        .def("getLayer", [](const TSinkhornSolverBase *a) {
            return a->layer;
            },"Return current layer index.")
        .def("getError",[](TSinkhornSolverBase *SinkhornSolver) {
            double result;
            SinkhornSolver->getError(&result);
            return result;
        },"Return current error (used as stopping criterion).");
        

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverStandard,TSinkhornSolverBase>(m, "TSinkhornSolverStandard",
            "Solver class for standard entropic optimal transport problems with fixed marginal constraints.")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                THierarchicalCostFunctionProvider *costProvider,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return new TSinkhornSolverStandard(
                    MultiScaleSetupX->nLayers,
                    epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                    layerCoarsest, layerFinest,
                    errorGoal,
                    cfg,
                    MultiScaleSetupX->HP, MultiScaleSetupY->HP,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    costProvider
                    );
            }),
            py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
            py::arg("MultiScaleSetupX"), py::arg("MultiScaleSetupY"), py::arg("costProvider"),
            py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault,
            R"(
            Args:
                epsScalingHandler: instance of TEpsScalingHandler to control epsilon schedule
                layerCoarsest: coarsest layer at which to start solving
                layerFinest: finest layer that should be solved
                errorGoal: the primary stopping criterion is reaching this threshold
                MultiScaleSetupX: TMultiScaleSetup instance describing first marginal
                MultiScaleSetupY: TMultiScaleSetup instance describing second marginal
                costProvider: THierarchicalCostFunctionProvider instance describing the cost function
                cfg: TSinkhornSolverParameters (optional, default values will be used if omitted))"
            )
        .def("getKernel",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return SinkhornSolver->kernel;
        })
        .def("getKernelCSRData",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return SinkhornKernelGetCSRData(SinkhornSolver->kernel);
        })
        .def("getKernelPosData",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return SinkhornKernelGetPosData(SinkhornSolver->kernel);
        })
        .def("getKernelCSRDataTuple",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return getSparseCSRDataTuple(SinkhornKernelGetCSRData(SinkhornSolver->kernel));
        })
        .def("getKernelPosDataTuple",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return getSparsePosDataTuple(SinkhornKernelGetPosData(SinkhornSolver->kernel));
        })
        .def("getScorePrimalUnreg",[](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->scorePrimalUnreg();
        })
        .def("getScoreTransportCost",[](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->scoreTransportCost();
        })
        .def("getKernelEntryCount", [](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->kernel.nonZeros();
        })
        .def("findKernelLine", [](TSinkhornSolverStandard * SinkhornSolver, int layerFinest, int a, int mode, double slack) {
            return findKernelLine(&(SinkhornSolver->kernelGenerator), layerFinest, a, mode, slack);
        }, py::arg("layerFinest"), py::arg("a"), py::arg("mode"), py::arg("slack"),
            R"(
            Args:
                layerFinest: index of desired layer
                a: index of kernel line to be determined
                mode: 0: rows, 1: columns
                slack: threshold below maximal value up to which entries are included
            Returns:
                A tuple of an int32 array and a double array, containing indices and effective cost values of dominating kernel entries.)"
        );
        

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverStandardLInf,TSinkhornSolverStandard>(m, "TSinkhornSolverStandardLInf")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                THierarchicalCostFunctionProvider *costProvider,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return new TSinkhornSolverStandardLInf(
                    MultiScaleSetupX->nLayers,
                    epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                    layerCoarsest, layerFinest,
                    errorGoal,
                    cfg,
                    MultiScaleSetupX->HP, MultiScaleSetupY->HP,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    costProvider
                    );
            }),
            py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
            py::arg("MultiScaleSetupX"), py::arg("MultiScaleSetupY"), py::arg("costProvider"),
            py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault
            );

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverSoftMarginal,TSinkhornSolverStandard>(m, "TSinkhornSolverSoftMarginal")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                THierarchicalCostFunctionProvider *costProvider,
                TSoftMarginalFunction *FX, TSoftMarginalFunction *FY,
                double kappaX, double kappaY,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return new TSinkhornSolverSoftMarginal(
                    MultiScaleSetupX->nLayers,
                    epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                    layerCoarsest, layerFinest,
                    errorGoal,
                    cfg,
                    MultiScaleSetupX->HP, MultiScaleSetupY->HP,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    costProvider,
                    FX, FY,
                    kappaX, kappaY
                    );
            }),
            py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
            py::arg("MultiScaleSetupX"), py::arg("MultiScaleSetupY"), py::arg("costProvider"),
            py::arg("FX"), py::arg("FY"), py::arg("kappaX"), py::arg("kappaY"),
            py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault
            )
        .def("setKappa", (void (TSinkhornSolverSoftMarginal::*)(double)) &TSinkhornSolverSoftMarginal::setKappa)
        .def("setKappa", (void (TSinkhornSolverSoftMarginal::*)(double, double)) &TSinkhornSolverSoftMarginal::setKappa);
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverKLMarginals,TSinkhornSolverSoftMarginal>(m, "TSinkhornSolverKLMarginals")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                THierarchicalCostFunctionProvider *costProvider,
                double kappa,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return new TSinkhornSolverKLMarginals(
                    MultiScaleSetupX->nLayers,
                    epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                    layerCoarsest, layerFinest,
                    errorGoal,
                    cfg,
                    MultiScaleSetupX->HP, MultiScaleSetupY->HP,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->muH, MultiScaleSetupY->muH,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    costProvider,
                    kappa
                    );
            }),
            py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
            py::arg("MultiScaleSetupX"), py::arg("MultiScaleSetupY"), py::arg("costProvider"),
            py::arg("kappa"),
            py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault
            );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSoftMarginalFunction>(m, "TSoftMarginalFunction")
        .def("f",[](const TSoftMarginalFunction &a, py::array_t<double> &muEff, py::array_t<double> &mu, const double kappa) {
                py::buffer_info muBuffer=mu.request();
                py::buffer_info muEffBuffer=muEff.request();
                
                int res=muBuffer.shape[0];
                return a.f((double*) muEffBuffer.ptr, (double*) muBuffer.ptr, kappa, res);
        });
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSoftMarginalFunctionKL,TSoftMarginalFunction>(m, "TSoftMarginalFunctionKL")
        .def(py::init([]() {return new TSoftMarginalFunctionKL();}))
        .def(py::init([](double muThresh) {return new TSoftMarginalFunctionKL(muThresh);}));
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSoftMarginalFunctionL1,TSoftMarginalFunction>(m, "TSoftMarginalFunctionL1")
        .def(py::init([]() {return new TSoftMarginalFunctionL1();}));
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverBarycenter,TSinkhornSolverBase>(m, "TSinkhornSolverBarycenter","Solver for optimal transport barycenter.")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetupBarycenterContainer *MultiScaleSetupBarycenterContainer,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return new TSinkhornSolverBarycenter(MultiScaleSetupBarycenterContainer->HPZ->nLayers,
                epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                layerCoarsest, layerFinest,
                errorGoal,
                cfg,
                MultiScaleSetupBarycenterContainer->nMarginals,
                MultiScaleSetupBarycenterContainer->weights,
                MultiScaleSetupBarycenterContainer->HP, MultiScaleSetupBarycenterContainer->HPZ,
                MultiScaleSetupBarycenterContainer->muH, MultiScaleSetupBarycenterContainer->muZH,
                MultiScaleSetupBarycenterContainer->alphaH, MultiScaleSetupBarycenterContainer->betaH,
                MultiScaleSetupBarycenterContainer->costProvider
                );
        }),
        py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
        py::arg("MultiScaleSetupBarycenterContainer"),
        py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault
        )
        .def("getCostFunctionProvider",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return SinkhornSolver->costProvider[n];
        })
        .def("getKernel",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return SinkhornSolver->kernel[n];
        })
        .def("getKernelCSRData",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return SinkhornKernelGetCSRData(SinkhornSolver->kernel[n]);
        })
        .def("getKernelPosData",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return SinkhornKernelGetPosData(SinkhornSolver->kernel[n]);
        })
        .def("getKernelCSRDataTuple",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return getSparseCSRDataTuple(SinkhornKernelGetCSRData(SinkhornSolver->kernel[n]));
        })
        .def("getKernelPosDataTuple",[](const TSinkhornSolverBarycenter * const SinkhornSolver, int n) {
            return getSparsePosDataTuple(SinkhornKernelGetPosData(SinkhornSolver->kernel[n]));
        })
        .def("getU", [](const TSinkhornSolverBarycenter * const a, const int nMarginal) {
            return getArrayFromRaw<double>(a->u[nMarginal].data(), a->u[nMarginal].size());
            })
        .def("getV", [](const TSinkhornSolverBarycenter * const a, const int nMarginal) {
            return getArrayFromRaw<double>(a->v[nMarginal].data(), a->v[nMarginal].size());
            })
        .def("findKernelLine", [](TSinkhornSolverBarycenter * SinkhornSolver, const int nMarginal, int layerFinest, int a, int mode, double slack) {
            return findKernelLine(SinkhornSolver->kernelGenerator[nMarginal],layerFinest, a, mode, slack);
        }, py::arg("nMarginal"), py::arg("layerFinest"), py::arg("a"), py::arg("mode"), py::arg("slack"),
            R"(
            Args:
                nMarginal: index of reference marginal
                layerFinest: index of desired layer
                a: index of kernel line to be determined
                mode: 0: rows, 1: columns
                slack: threshold below maximal value up to which entries are included)"
        );


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSinkhornSolverBarycenterKLMarginals,TSinkhornSolverBarycenter>(m, "TSinkhornSolverBarycenterKLMarginals", "Solver for optimal transport barycenter with KL soft marginals.")
        .def(py::init([](TEpsScalingHandler *epsScalingHandler,
                int layerCoarsest, int layerFinest,
                double errorGoal,
                TMultiScaleSetupBarycenterContainer *MultiScaleSetupBarycenterContainer,
                double kappa,
                TSinkhornSolverBase::TSinkhornSolverParameters cfg
                ) {
            return TSinkhornSolverBarycenterKLMarginals(MultiScaleSetupBarycenterContainer->HPZ->nLayers,
                epsScalingHandler->nEpsLists, epsScalingHandler->epsLists,
                layerCoarsest, layerFinest,
                errorGoal,
                cfg,
                MultiScaleSetupBarycenterContainer->nMarginals,
                MultiScaleSetupBarycenterContainer->weights,
                MultiScaleSetupBarycenterContainer->HP, MultiScaleSetupBarycenterContainer->HPZ,
                MultiScaleSetupBarycenterContainer->muH, MultiScaleSetupBarycenterContainer->muZH,
                MultiScaleSetupBarycenterContainer->alphaH, MultiScaleSetupBarycenterContainer->betaH,
                MultiScaleSetupBarycenterContainer->costProvider,
                kappa
                );
        }),
        py::arg("epsScalingHandler"), py::arg("layerCoarsest"), py::arg("layerFinest"), py::arg("errorGoal"),
        py::arg("MultiScaleSetupBarycenterContainer"),
        py::arg("kappa"),
        py::arg("cfg")=TSinkhornSolverBase::SinkhornSolverParametersDefault
        )
        .def("getScorePrimal", [](TSinkhornSolverBarycenterKLMarginals *SinkhornSolver) {
            double result;
            SinkhornSolver->getScorePrimal(&result);
            return result;
        })
        .def_readwrite("kappa", &TSinkhornSolverBarycenterKLMarginals::kappa);

    #endif // Sinkhorn
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


    py::class_<TGeometry_Euclidean>(m, "TGeometry_Euclidean")
         .def(py::init([]() {
            return TGeometry_Euclidean();
        }),"Class describing flat Euclidean geometry. Experimental.");


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    m.def("interpolateEuclidean",
            [](TSparsePosContainer &couplingData, py::array_t<double> &posX, py::array_t<double> &posY, double t) {
                py::buffer_info bufferX = posX.request();
                py::buffer_info bufferY = posY.request();
                return PyInterpolateEuclidean(couplingData, (double*) bufferX.ptr, (double*) bufferY.ptr, bufferX.shape[1], t);
            },
            py::arg("couplingData"),
            py::arg("posX"), py::arg("posY"),
            py::arg("t"),
            R"(
            Compute displacement interpolation for Wasserstein-p distance in Euclidean space.
            
            Args:
                couplingData: TSparsePosContainer containing non-zero entries of coupling in sparse POS format
                posX: 2d numpy.float64 array containing positions of first marginal points
                posY: 2d numpy.float64 array containing positions of second marginal points
                t: float in [0,1], gives time at which to compute interpolation. t=0: first marginal, t=1: second marginal, t=0.5: midpoint.
            )"
            );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  
    m.def("interpolateEuclideanWFR",
            [](TSparsePosContainer &couplingData,
                    py::array_t<double> &muXEff, py::array_t<double> &muYEff,
                    py::array_t<double> &muX, py::array_t<double> &muY,
                    py::array_t<double> &posX, py::array_t<double> &posY,
                    double t, double WFScale) {
                py::buffer_info bufferX = posX.request();
                py::buffer_info bufferY = posY.request();
                return PyInterpolateEuclideanWFR(couplingData,
                        getDataPointerFromNumpyArray<double>(muXEff),
                        getDataPointerFromNumpyArray<double>(muYEff),
                        getDataPointerFromNumpyArray<double>(muX),
                        getDataPointerFromNumpyArray<double>(muY),
                        (double*) bufferX.ptr, (double*) bufferY.ptr,
                        bufferX.shape[1], t, WFScale);
            },
            py::arg("couplingData"),
            py::arg("muXEff"), py::arg("muYEff"),
            py::arg("muX"), py::arg("muY"),
            py::arg("posX"), py::arg("posY"),
            py::arg("t"), py::arg("WFScale"),
            R"(
            Compute displacement interpolation for Wasserstein--Fisher--Rao distance in Euclidean space.
            
            Args:
                couplingData: TSparsePosContainer containing non-zero entries of coupling in sparse POS format
                muXEff: first marginal of coupling (which can be different from input measure on first marginal)
                muYEff: second marginal of coupling
                muX: input measure on first marginal
                muY: input measure on second marginal
                posX: 2d numpy.float64 array containing positions of first marginal points
                posY: 2d numpy.float64 array containing positions of second marginal points
                t: float in [0,1], gives time at which to compute interpolation. t=0: first marginal, t=1: second marginal, t=0.5: midpoint.
                WFScale: scale parameter determining trade-off between transport and mass change. Maximal travelling distance is given by pi/2*WFScale.)"
            );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    m.def("projectInterpolation",
            [](TParticleContainer &particles, py::array_t<double> &img) {
                TDoubleMatrix *imgMat=getMatrixFromNumpyArray<double>(img);
                PyProjectInterpolation(particles,imgMat);
                delete imgMat;
            },
            py::arg("particles"), py::arg("img"),
            R"(
            Projects a TParticleContainer object (usually storing a displacement interpolation) to a grid via bi-linear interpolation.
            
            Args:
                particles: TParticleContainer storing positions and locations of a list of particles in d dimensions
                img: d-dimensional numpy.float64 array to which the particles should be rasterized

            The mass of each particle is split to the nearest pixels of img, weighted by the relative position along each axis.)");
            
 

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    m.def("getCTransform",
            [](TMultiScaleSetup *MultiScaleSetupX, TMultiScaleSetup *MultiScaleSetupY, THierarchicalCostFunctionProvider *costProvider,
                    int layerFinest, int mode) {


                THierarchicalDualMaximizer::getMaxDual(MultiScaleSetupX->HP, MultiScaleSetupY->HP,
                        MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH, layerFinest,
                        costProvider,
                        mode);
            },"Compute c-transform (or hierarchical approximation thereof). Experimental.");
            
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    m.def("setVerboseMode", [](bool verbose) { verbose_mode=verbose; }, "Set verbose mode",
            py::arg("verbose"));
    


}
