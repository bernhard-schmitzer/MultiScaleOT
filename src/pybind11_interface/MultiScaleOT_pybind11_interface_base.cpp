#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Common/Verbose.h>

#include<pybind11/include/pybind11/pybind11.h>
#include<pybind11/include/pybind11/numpy.h>

#include"MultiScaleOT_pybind11_interface_common.h"

namespace py = pybind11;


using namespace std;

#ifdef USE_SINKHORN
void init_sinkhorn(py::module_ &);
#endif
#ifdef USE_CPLEX
void init_cplex(py::module_ &);
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


TParticleContainer PyInterpolateEuclideanHK(
        TSparsePosContainer &couplingData,
        double *muXEff, double *muYEff,
        double *muX, double *muY,
        double *posX, double *posY, int dim, double t, double HKscale) {
 
    TGeometry_Euclidean geometry;
        
    TParticleContainer result=ModelHK_Interpolate<TGeometry_Euclidean>(
            couplingData,
            muXEff, muYEff, muX, muY,
            posX, posY,
            dim, t, HKscale, geometry);

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
    py::class_<TSparseCSRContainer>(m,"TSparseCSRContainer","Sparse container for coupling data, using the scipy.sparse.csr_matrix format.")
        .def(py::init([](py::array_t<double> &data, py::array_t<int> &indices, py::array_t<int> &indptr,
                const int xres, const int yres) {
            return getSparseCSRContainerFromData(data,indices,indptr,xres,yres);
            }),
            py::arg("data"),py::arg("indices"),py::arg("indptr"),py::arg("xres"),py::arg("yres")
            )
        .def("getDataTuple",[](const TSparseCSRContainer &csrContainer) {
            return getSparseCSRDataTuple(csrContainer);
        },"Return list of non-zero entries, column indices, and row separation indices.");


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TSparsePosContainer>(m,"TSparsePosContainer","Sparse container for coupling data, using the scipy.sparse.coo_matrix format.")
        .def(py::init([](py::array_t<double> &mass, py::array_t<int> &posX, py::array_t<int> &posY,
                const int xres, const int yres) {
            return getSparsePosContainerFromData(mass,posX,posY,xres,yres);
            }),
            py::arg("mass"),py::arg("posX"),py::arg("posY"),py::arg("xres"),py::arg("yres")
            )
        .def("getDataTuple",[](const TSparsePosContainer &posContainer) {
            return getSparsePosDataTuple(posContainer);
        },"Return list of non-zero values, row and column indices.");
 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<TParticleContainer>(m,"TParticleContainer","Container for list of mass particles, contains an array of masses and an array for locations.")
        .def(py::init([](py::array_t<double> &mass, py::array_t<double> &pos) {
            return getParticleContainerFromData(mass,pos);
            }),
            py::arg("mass"),py::arg("pos"),
            R"(
            Args:
                mu: 1d double array of particle masses
                pos: 2d double array of particle locations)"
            )
        .def("getDataTuple",[](const TParticleContainer &particleContainer) {
            return getParticleDataTuple(particleContainer);
        },"Return list of masses and positions.");

    
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
        .def("coarsenSignal", [](TMultiScaleSetup *a, py::array_t<double> &signal, const int lFinest, const int lCoarsest, const int mode) {
            vector<double*> signalH(lFinest-lCoarsest+1);
            py::list result;
            for(int i=lCoarsest;i<lFinest;i++) {
                py::array_t<double> *layerArray=new py::array_t<double>(a->HP->layers[i]->nCells);
                result.append(layerArray);
                py::buffer_info layerArrayBuffer=layerArray->request();
                signalH[i-lCoarsest]=(double*) layerArrayBuffer.ptr;
            }
            result.append(signal);
            py::buffer_info signalBuffer=signal.request();
            signalH[lFinest-lCoarsest]=(double*) signalBuffer.ptr;
            a->HP->signal_propagate_double(signalH.data(), lCoarsest, lFinest, mode);

            return result;

            },py::arg("signal"),py::arg("lFinest"),py::arg("lCoarsest"),py::arg("mode"),
            R"(
            Coarsens a double-valued signal from one layer to subsequent coarser layers.
            
            Args:
                signal: 1d double array with signal at finest layer
                lFinest: id of finest layer
                lCoarsest: id of coarsest layer
                mode: determines how signal is refined.

                    * 0: coarse value is min over children
                    * 1: coarse value is max over children

            Returns:
                list of double array containing coarsened signal at requested layers.)")
        .def("coarsenSignalInt", [](TMultiScaleSetup *a, py::array_t<int> &signal, const int lFinest, const int lCoarsest, const int mode) {
            vector<int*> signalH(lFinest-lCoarsest+1);
            py::list result;
            for(int i=lCoarsest;i<lFinest;i++) {
                py::array_t<int> *layerArray=new py::array_t<int>(a->HP->layers[i]->nCells);
                result.append(layerArray);
                py::buffer_info layerArrayBuffer=layerArray->request();
                signalH[i-lCoarsest]=(int*) layerArrayBuffer.ptr;
            }
            result.append(signal);
            py::buffer_info signalBuffer=signal.request();
            signalH[lFinest-lCoarsest]=(int*) signalBuffer.ptr;
            a->HP->signal_propagate_int(signalH.data(), lCoarsest, lFinest, mode);

            return result;

            },py::arg("signal"),py::arg("lFinest"),py::arg("lCoarsest"),py::arg("mode"),
            R"(
            Coarsens a int32-valued signal from one layer to subsequent coarser layers.
            
            Args:
                signal: 1d int32 array with signal at finest layer
                lFinest: id of finest layer
                lCoarsest: id of coarsest layer
                mode: determines how signal is refined.

                    * 0: coarse value is min over children
                    * 1: coarse value is max over children

            Returns:
                list of double array containing coarsened signal at requested layers.)")
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
                1d double array containing refined signal at fine layer (one layer below coarse layer))")
        .def("updatePositions", [](TMultiScaleSetup *a, py::array_t<double> &newPositions) {
            py::buffer_info posBuffer=newPositions.request();
            
            const int xres=a->HP->layers[a->HP->nLayers-1]->nCells;
            
            
            if(xres!=posBuffer.shape[0]) throw std::invalid_argument("number of new points does not match finest layer size.");
            if(a->HP->dim!=posBuffer.shape[1]) throw std::invalid_argument("dimension of new points does not match previous dimension");

            return a->UpdatePositions((double*) posBuffer.ptr);
            },
            py::arg("newPositions"),
            R"(
            Overwrites the spatial positions of the multi scale representation at the finest layer and internally updates all coarser nodes and radii values.)")
        .def("updateMeasure", [](TMultiScaleSetup *a, py::array_t<double> &newMeasure) {
            py::buffer_info measureBuffer=newMeasure.request();
            
            const int xres=a->HP->layers[a->HP->nLayers-1]->nCells;
            
            
            if(xres!=measureBuffer.shape[0]) throw std::invalid_argument("size of new measure does not match finest layer size.");

            return a->UpdateMeasure((double*) measureBuffer.ptr);
            },
            py::arg("newMeasure"),
            R"(
            Overwrites the point masses of the multi scale representation at the finest layer and internally updates all coarser nodes.)");

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
                bool HKmode, double HKscale) {
            return new THierarchicalCostFunctionProvider_SquaredEuclidean(
                     MultiScaleSetupX->posH, MultiScaleSetupY->posH,
                    MultiScaleSetupX->radii, MultiScaleSetupY->radii,
                    MultiScaleSetupX->dim, MultiScaleSetupX->nLayers-1,
                    true,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    1.,
                    HKmode, HKscale
                    );
        }),
            py::arg("MultiScaleSetupX"),py::arg("MultiScaleSetupY"),py::arg("HKmode")=false,py::arg("HKscale")=1.,
            R"(
            Args:
                MultiScaleSetupX: TMultiScaleSetup instance describing first marginal
                MultiScaleSetupY: TMultiScaleSetup instance describing second marginal
                HKmode: whether Hellinger--Kantorovich mode should be activated
                HKscale: trade-off weight between transport and mass change. Maximal transport distance is HKscale*pi/2.
            )"
        )
        //////////////////////////////////////////////////////
        .def_readwrite("HKmode", &THierarchicalCostFunctionProvider_SquaredEuclidean::HKmode)
        .def_readwrite("layerBottom", &THierarchicalCostFunctionProvider_SquaredEuclidean::layerBottom)
        .def("setScale",&THierarchicalCostFunctionProvider_SquaredEuclidean::setScale)
        .def("setHKscale",&THierarchicalCostFunctionProvider_SquaredEuclidean::setHKscale);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    py::class_<THierarchicalCostFunctionProvider_PEuclidean, THierarchicalCostFunctionProvider>
            (m, "THierarchicalCostFunctionProvider_PEuclidean","Represents the Euclidean distance cost function to power p.")
        //////////////////////////////////////////////////////
        .def(py::init([](
                TMultiScaleSetup *MultiScaleSetupX,
                TMultiScaleSetup *MultiScaleSetupY,
                double p) {
            return new THierarchicalCostFunctionProvider_PEuclidean(
                     MultiScaleSetupX->posH, MultiScaleSetupY->posH,
                    MultiScaleSetupX->radii, MultiScaleSetupY->radii,
                    MultiScaleSetupX->dim, MultiScaleSetupX->nLayers-1,
                    true,
                    MultiScaleSetupX->alphaH, MultiScaleSetupY->alphaH,
                    1.,
                    p
                    );
        }),
            py::arg("MultiScaleSetupX"),py::arg("MultiScaleSetupY"),py::arg("p")=2.,
            R"(
            Args:
                MultiScaleSetupX: TMultiScaleSetup instance describing first marginal
                MultiScaleSetupY: TMultiScaleSetup instance describing second marginal
                p: exponent for distance
            )"
        )
        //////////////////////////////////////////////////////
        .def_readwrite("p", &THierarchicalCostFunctionProvider_PEuclidean::p);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
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
  
    m.def("interpolateEuclideanHK",
            [](TSparsePosContainer &couplingData,
                    py::array_t<double> &muXEff, py::array_t<double> &muYEff,
                    py::array_t<double> &muX, py::array_t<double> &muY,
                    py::array_t<double> &posX, py::array_t<double> &posY,
                    double t, double HKscale) {
                py::buffer_info bufferX = posX.request();
                py::buffer_info bufferY = posY.request();
                return PyInterpolateEuclideanHK(couplingData,
                        getDataPointerFromNumpyArray<double>(muXEff),
                        getDataPointerFromNumpyArray<double>(muYEff),
                        getDataPointerFromNumpyArray<double>(muX),
                        getDataPointerFromNumpyArray<double>(muY),
                        (double*) bufferX.ptr, (double*) bufferY.ptr,
                        bufferX.shape[1], t, HKscale);
            },
            py::arg("couplingData"),
            py::arg("muXEff"), py::arg("muYEff"),
            py::arg("muX"), py::arg("muY"),
            py::arg("posX"), py::arg("posY"),
            py::arg("t"), py::arg("HKscale"),
            R"(
            Compute displacement interpolation for Hellinger--Kantorovich distance in Euclidean space.
            
            Args:
                couplingData: TSparsePosContainer containing non-zero entries of coupling in sparse POS format
                muXEff: first marginal of coupling (which can be different from input measure on first marginal)
                muYEff: second marginal of coupling
                muX: input measure on first marginal
                muY: input measure on second marginal
                posX: 2d numpy.float64 array containing positions of first marginal points
                posY: 2d numpy.float64 array containing positions of second marginal points
                t: float in [0,1], gives time at which to compute interpolation. t=0: first marginal, t=1: second marginal, t=0.5: midpoint.
                HKscale: scale parameter determining trade-off between transport and mass change. Maximal travelling distance is given by pi/2*HKscale.)"
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
    
    #ifdef USE_SINKHORN
    init_sinkhorn(m);
    #endif
    #ifdef USE_CPLEX
    init_cplex(m);
    #endif


}
