#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Common/Verbose.h>

#ifdef USE_SINKHORN
#include<Sinkhorn.h>
#endif

#include<pybind11/include/pybind11/pybind11.h>
#include<pybind11/include/pybind11/numpy.h>

#include"MultiScaleOT_pybind11_interface_common.h"

namespace py = pybind11;


using namespace std;

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


#ifdef USE_SINKHORN
void init_sinkhorn(py::module_ &m) {

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        },"Returns a tuple of numpy arrays (mass,posStart,posEnd) of types (float64,int32,int32) containing the non-zero kernel entries in sparse POS format (row indices, column indices, mass values).");

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


    py::class_<TSinkhornSolverBase>(m,"TSinkhornSolverBase",
            R"(
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
        },"Return a TKernelMatrix object holding the current stabilized kernel.")
        .def("getKernelCSRData",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return SinkhornKernelGetCSRData(SinkhornSolver->kernel);
        },"Returns a TSparseCSRContainer object containing the non-zero stabilized kernel entries in sparse CSR format.")
        .def("getKernelPosData",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return SinkhornKernelGetPosData(SinkhornSolver->kernel);
        },"Returns a TSparsePosContainer object containing the non-zero stabilized kernel entries in sparse POS format.")
        .def("getKernelCSRDataTuple",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return getSparseCSRDataTuple(SinkhornKernelGetCSRData(SinkhornSolver->kernel));
        },"Returns a tuple of numpy arrays (data,indices,indptr) of types (float64,int32,int32) containing the non-zero stabilized kernel entries in sparse CSR format (see scipy.sparse.csr_matrix).")
        .def("getKernelPosDataTuple",[](const TSinkhornSolverStandard * const SinkhornSolver) {
            return getSparsePosDataTuple(SinkhornKernelGetPosData(SinkhornSolver->kernel));
        },"Returns a tuple of numpy arrays (mass,posStart,posEnd) of types (float64,int32,int32) containing the non-zero stabilized kernel entries in sparse POS format (mass values, row indices, column indices, see also scipy.sparse.coo_matrix).")
        .def("getScorePrimalUnreg",[](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->scorePrimalUnreg();
        },"Return current primal cost without the entropy term (only transport term and marginal discrepancies in unbalanced cases).")
        .def("getScoreTransportCost",[](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->scoreTransportCost();
        },"Return transport cost of current coupling (i.e. integral of coupling against cost function on product space.")
        .def("getKernelEntryCount", [](TSinkhornSolverStandard * SinkhornSolver) {
            return SinkhornSolver->kernel.nonZeros();
        },"Returns number of non-zero entries in current stabilized kernel.")
        .def("getMarginalX", [](const TSinkhornSolverStandard * const a) {
            TMarginalVector marg=a->getMarginalX();
            return getArrayFromRaw<double>(marg.data(), marg.size());
            },"Return 1st marginal of current coupling.")
        .def("getMarginalY", [](const TSinkhornSolverStandard * const a) {
            TMarginalVector marg=a->getMarginalY();
            return getArrayFromRaw<double>(marg.data(), marg.size());
            },"Return 1st marginal of current coupling.")
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
        .def("getMarginalX", [](const TSinkhornSolverBarycenter * const a, const int nMarginal) {
            TMarginalVector marg=a->getMarginalX(nMarginal);
            return getArrayFromRaw<double>(marg.data(), marg.size());
            },"Return 1st marginal of current coupling between nMarginal-th reference measure and current barycenter candidate.")
        .def("getMarginalY", [](const TSinkhornSolverBarycenter * const a, const int nMarginal) {
            TMarginalVector marg=a->getMarginalY(nMarginal);
            return getArrayFromRaw<double>(marg.data(), marg.size());
            },"Return 2nd marginal of current coupling between nMarginal-th reference measure and current barycenter candidate.")
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


}

#endif // Sinkhorn

