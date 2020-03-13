#ifndef TSinkhornSolver_H_
#define TSinkhornSolver_H_

#include<Common/ErrorCodes.h>
#include<Common/Tools.h>
#include<Common/Verbose.h>

#include<Common/THierarchicalPartition.h>
#include<Sinkhorn/TSinkhornKernel.h>

// abstract base class for sparse multi-scale sinkhorn type algorithm with eps scaling
// only implements abstract high level functions, such as:
//	iterate over scales
//	for each scale iterate over eps list
//	for each scale, eps: solve
//	during solve: check for absorption and errors

class TSinkhornSolverBase {
public:

	static constexpr int MSG_ABSORB_REITERATE=1;
	static constexpr int MSG_EXCEEDMAXITERATIONS=30101;
	static constexpr int MSG_NANSCALING=30102;	
	static constexpr int MSG_NANINERROR=30103;	
	static constexpr int MSG_ABSORB_TOOMANYABSORPTIONS=30201;

	// summarize "more detailed" configuration parameters into one object
	struct TSinkhornSolverParameters {
		int maxIterations; // maximal iterations per singleSolve
		int innerIterations; // nr of iterations between absorption and error checks
		int maxAbsorptionLoops; // nr of maximal loops of absorption until algorithm must stabilize
		double absorption_scalingBound; // maximal value of a scaling variable
		double absorption_scalingLowerBound; // value above which we do a safety absorption // NOT YET IMPLEMENTED!
		double truncation_thresh; // truncation threshold in kernel sparsification
		bool refineKernel; // whether initial kernel should be generated via refinement on subsequent layers
		bool generateFinalKernel; // whether to automatically generate a final up-to-date kernel after successful solving
		TSinkhornSolverParameters(int _maxIterations, int _innerIterations, int _maxAbsorptionLoops,
				double _absorption_scalingBound, double _absorption_scalingLowerBound, double _truncation_thresh, bool _refineKernel,
				bool _generateFinalKernel) :
				maxIterations(_maxIterations), innerIterations(_innerIterations), maxAbsorptionLoops(_maxAbsorptionLoops),
				absorption_scalingBound(_absorption_scalingBound), absorption_scalingLowerBound(_absorption_scalingLowerBound),
				truncation_thresh(_truncation_thresh), refineKernel(_refineKernel), generateFinalKernel(_generateFinalKernel) {};
		TSinkhornSolverParameters() {};
				
	};
	static TSinkhornSolverParameters SinkhornSolverParametersDefault;
	
//	static constexpr TSinkhornSolverParameters SinkhornSolverParametersDefault={100000,100,100,1E3,1E3,1E-20,false};


	TSinkhornSolverParameters cfg; // store all configuration parameters
	double errorGoal; // error accuracy to be achieved

	int nLayers; // total number of hierarchy layers
	int *nEpsList; // number of eps at each level
	double **epsLists; // eps lists for each level
	int layerCoarsest, layerFinest; // numbers of first and final layer to be solved

	int layer; // currently active layer
	double eps; // currently active eps
	
	bool kernelValid; // keeps track whether the currently held kernel matches the current problem setup (eps, absorption status etc.)


	TSinkhornSolverBase(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg
		);
		

	virtual int initialize();
	virtual int changeEps(const double newEps);
	virtual int refineDuals(__attribute__((unused)) const int newLayer) { return ERR_BASE_NOTIMPLEMENTED; };
	virtual int changeLayer(const int newLayer);
	virtual void changeParameters(TSinkhornSolverParameters newCfg);


	virtual int iterate(__attribute__((unused)) const int n) { return ERR_BASE_NOTIMPLEMENTED; };
	virtual int checkAbsorb(__attribute__((unused)) const double maxValue) { return ERR_BASE_NOTIMPLEMENTED; };
	virtual int absorb() { return ERR_BASE_NOTIMPLEMENTED; };
	virtual int getError(__attribute__((unused)) double * const result) { return ERR_BASE_NOTIMPLEMENTED; };
	virtual int generateKernel() { return ERR_BASE_NOTIMPLEMENTED; };
	
	virtual int refineKernel() { return ERR_BASE_NOTIMPLEMENTED; };
	
	
	int solveSingle();
	int solveLayer();
	int solve();
	
	
	virtual double scorePrimalUnreg() { return 0.; }; // careful: just a dummy
	
};

// class for default 2-marginal transport problem with fixed marginal constraints
class TSinkhornSolverStandard : public TSinkhornSolverBase {
public:

	// objects that are given from outside
	THierarchicalPartition *HPX, *HPY;
	double **muXH, **muYH;
	// reference measures for entropy regularization
	double **rhoXH, **rhoYH;
	THierarchicalCostFunctionProvider *costProvider;
	double **alphaH, **betaH;

	// objects the algorithm creates
	TSinkhornKernelGenerator kernelGenerator;
	TKernelMatrix kernel, kernelT;
	int xres,yres;
	TMarginalVector u,v;

	// some info on current layer
	double *muX,*muY,*alpha,*beta;


	TSinkhornSolverStandard(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_rhoXH, double **_rhoYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider
		);

	TSinkhornSolverStandard(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider
		) : TSinkhornSolverStandard(
				_nLayers,
				_nEpsList,
				_epsLists,
				_layerCoarsest, _layerFinest,
				_errorGoal,
				_cfg,
				_HPX, _HPY,
				_muXH, _muYH,
				_muXH, _muYH,
				_alphaH, _betaH,		
				_costProvider
				) {};



	virtual int initialize();
	virtual int refineDuals(const int newLayer);
	virtual int changeLayer(const int newLayer);

	virtual int checkAbsorb(const double maxValue);
	virtual int absorb();
	virtual int generateKernel();
	virtual int refineKernel();
	

	// model specific
	virtual int iterate(const int n);
	virtual int getError(double * const result);
	virtual double scorePrimalUnreg();

	virtual double scoreTransportCost();
	
	std::vector<double> getMarginalX();
	std::vector<double> getMarginalY();
	void writeMarginalX(double *buffer);
	void writeMarginalY(double *buffer);

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// class for default 2-marginal transport problem with fixed marginal constraints, with LInf error
class TSinkhornSolverStandardLInf : public TSinkhornSolverStandard {
public:



	TSinkhornSolverStandardLInf(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_rhoXH, double **_rhoYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider
		) : TSinkhornSolverStandard(
		_nLayers, _nEpsList, _epsLists,
		_layerCoarsest, _layerFinest, _errorGoal, _cfg,
		_HPX, _HPY, _muXH, _muYH, _rhoXH, _rhoYH, _alphaH, _betaH,		
		_costProvider) {};

	TSinkhornSolverStandardLInf(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider
		) : TSinkhornSolverStandardLInf(
				_nLayers,
				_nEpsList,
				_epsLists,
				_layerCoarsest, _layerFinest,
				_errorGoal,
				_cfg,
				_HPX, _HPY,
				_muXH, _muYH,
				_muXH, _muYH,
				_alphaH, _betaH,		
				_costProvider
				) {};



	// model specific
	virtual int getError(double * const result);

};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


// soft marginal constraints
// first: auxiliary classes to model the soft marginal objective functions
class TSoftMarginalFunction {
public:
	virtual void proxdiv(double *u, double *conv, double *mu, double *alpha, double kappa, double eps, int res)=0;
	virtual double PDGapContribution(double *muEff, double *alpha, double *mu, double kappa, int res)=0;
	virtual double f(const double * const muEff, const double * const mu, const double kappa, const int res) const =0;
	virtual ~TSoftMarginalFunction();
};

class TSoftMarginalFunctionKL : public TSoftMarginalFunction {
public:
	double muThresh;
	TSoftMarginalFunctionKL() : muThresh(0) {};
	TSoftMarginalFunctionKL(double _muThresh) : muThresh(_muThresh) {};
	virtual void proxdiv(double *u, double *conv, double *mu, double *alpha, double kappa, double eps, int res);
	virtual double PDGapContribution(double *muEff, double *alpha, double *mu, double kappa, int res);
	virtual double f(const double * const muEff, const double * const mu, const double kappa, const int res) const;
	virtual double fDual(double *alpha, double *mu, double kappa, int res);
};

class TSoftMarginalFunctionL1 : public TSoftMarginalFunction {
public:
	TSoftMarginalFunctionL1() {};
	virtual void proxdiv(double *u, double *conv, double *mu, double *alpha, double kappa, double eps, int res);
	virtual double PDGapContribution(double *muEff, double *alpha, double *mu, double kappa, int res);
	virtual double f(const double * const muEff, const double * const mu, const double kappa, const int res) const;
	virtual double fDual(double *alpha, double *mu, double kappa, int res);
};


// class for 2-marginal transport problem with soft marginal constraints
// weight of marginal terms is kappaX and kappaY
class TSinkhornSolverSoftMarginal : public TSinkhornSolverStandard {
public:

	static constexpr double DBL_INFINITY=1E100; // effective value for infinity
	TSoftMarginalFunction *FX, *FY;
	double kappaX, kappaY;

	TSinkhornSolverSoftMarginal(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_rhoXH, double **_rhoYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider,
		TSoftMarginalFunction *_FX, TSoftMarginalFunction *_FY,
		double _kappaX, double _kappaY
		);

	virtual int iterate(const int n);
	virtual int getError(double * const result);
	virtual double scorePrimalUnreg();
	void setKappa(double _kappa);
	void setKappa(double _kappaX, double _kappaY);

};

class TSinkhornSolverKLMarginals : public TSinkhornSolverSoftMarginal {
public:
	TSinkhornSolverKLMarginals(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_rhoXH, double **_rhoYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider,
		double _kappa
		);

	TSinkhornSolverKLMarginals(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg,
		THierarchicalPartition *_HPX, THierarchicalPartition *_HPY,
		double **_muXH, double **_muYH,
		double **_alphaH, double **_betaH,		
		THierarchicalCostFunctionProvider *_costProvider,
		double _kappa
		) : TSinkhornSolverKLMarginals(_nLayers,_nEpsList,_epsLists,_layerCoarsest,_layerFinest,
				_errorGoal,_cfg,_HPX,_HPY,_muXH,_muYH,_muXH,_muYH,_alphaH,_betaH,_costProvider,_kappa) {};


	~TSinkhornSolverKLMarginals();

	static double KL(const double * const muEff, const double * const mu, const double kappa, const int res, const double muThresh);
	static double KL(const double * const muEff, const double * const mu, const double kappa, const int res) { return KL(muEff, mu, kappa, res, 0); };
	static double KLDual(double *alpha, double *mu, double kappa, int res);

};

#endif
