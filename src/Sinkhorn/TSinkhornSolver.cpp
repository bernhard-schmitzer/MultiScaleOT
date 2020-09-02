#include"TSinkhornSolver.h"

//TSinkhornSolverBase::SinkhornSolverParametersDefault=TSinkhornSolverBase::TSinkhornSolverParameters(100000,100,100,1E3,1E3,1E-20,false);
TSinkhornSolverBase::TSinkhornSolverParameters TSinkhornSolverBase::SinkhornSolverParametersDefault(100000,100,100,1E3,1E3,1E-20,false,true);
//TSinkhornSolverBase::TSinkhornSolverParameters(100000,100,100,1E3,1E3,1E-20,false);

TSinkhornSolverBase::TSinkhornSolverBase(
		int _nLayers,
		int *_nEpsList,
		double **_epsLists,
		int _layerCoarsest, int _layerFinest,
		double _errorGoal,
		TSinkhornSolverParameters _cfg
		) {

	nLayers=_nLayers;
	nEpsList=_nEpsList;
	epsLists=_epsLists;
	layerCoarsest=_layerCoarsest;
	layerFinest=_layerFinest;
	
	cfg=_cfg;
	errorGoal=_errorGoal;
	
	kernelValid=false;
}

int TSinkhornSolverBase::initialize() {
	layer=layerCoarsest;
	kernelValid=false;
	return 0;
}

int TSinkhornSolverBase::changeEps(const double newEps) {
	eps=newEps;
	kernelValid=false;
	eprintf("\teps=%e\n",eps);
	return 0;
}

int TSinkhornSolverBase::changeLayer(const int newLayer) {
	layer=newLayer;
	kernelValid=false;
	eprintf("layer=%d\n",layer);
	return 0;
}

void TSinkhornSolverBase::changeParameters(TSinkhornSolverParameters newCfg) {
	cfg=newCfg;
	kernelValid=false;
}


int TSinkhornSolverBase::solveSingle() {
	int nIterations=0;
	int nAbsorptionLoops=0;
	
	int msg;
	double error;
	
	// compute first kernel
	if(!kernelValid) {
		msg=generateKernel();
		kernelValid=true;
		if(msg!=0) return msg;
	}
	
	
	while(true) {
		
		// inner iterations
		msg=iterate(cfg.innerIterations);
		
		// check if something went wrong during iterations
		if(msg!=0) return msg;
		
		// check if need to absorb
		msg=checkAbsorb(cfg.absorption_scalingBound);
		if(msg==MSG_ABSORB_REITERATE) {
			eprintf("\t\tabsorbing\n");
			// if absorption is required
			// increase counter of consecutive absorptions
			nAbsorptionLoops++;
			
			// check if too many consecutive absorptions already have happened
			if(nAbsorptionLoops>cfg.maxAbsorptionLoops) {
				return MSG_ABSORB_TOOMANYABSORPTIONS;
			}
			// otherwise, absorb
			msg=absorb();
			// if something went wrong during absorption
			if(msg!=0) return msg;
			
			// generate new kernel
			msg=generateKernel();
			if(msg!=0) return msg;
			kernelValid=true;

			
			// skip rest of this cycle and return to iterating
			continue;
		} else {
			// check if some other error occured
			if(msg!=0) return msg;
		}
		// if no absorption happened, reset consecutive absorption counter
		nAbsorptionLoops=0;
		
		// retrieve iteration accuracy error
		msg=getError(&error);
		if(msg!=0) return msg;
		
		eprintf("\t\terror: %e\n",error);
		if(error<=errorGoal) {
			// if numerical accuracy has been achieved without any errors, finish

			// check if "safety absorption" is recommended
			msg=checkAbsorb(cfg.absorption_scalingLowerBound);
			if(msg==MSG_ABSORB_REITERATE) {
				eprintf("\t\tsafety absorption.\n");
				// if another absorption is recommended (and in particular, some iterations thereafter)
				
				msg=absorb();
				// if something went wrong during absorption
				if(msg!=0) return msg;

				// generate new kernel
				msg=generateKernel();
				if(msg!=0) return msg;
				kernelValid=true;

				continue;

			} else {
				// otherwise, check for other errors
				if(msg!=0) return msg;			

				// then do a final absorption
				eprintf("\t\tfinal absorption\n");
				msg=absorb();
				// if something went wrong during absorption
				if(msg!=0) return msg;
				// note that due to absorption, kernel is no longer valid
				kernelValid=false;

				// return with no error code
				return 0;
			}

		}
		
		// increase iteration counter
		nIterations+=cfg.innerIterations;
		if(nIterations>cfg.maxIterations) {
			return MSG_EXCEEDMAXITERATIONS;
		}
	}
}

int TSinkhornSolverBase::solveLayer() {
	int msg;
	for(int nEps=0; nEps<nEpsList[layer]; nEps++) {
		changeEps(epsLists[layer][nEps]);
		// if we just began solving this layer (and we are not on finest layer, and cfg says so), then refine
		if(cfg.refineKernel && (nEps==0) && (layer>layerCoarsest)) {
			msg=refineKernel();
			if(msg!=0) return msg;
			kernelValid=true;
		}
		int msg=solveSingle();
		if(msg!=0) return msg;
	}
	return 0;
}

int TSinkhornSolverBase::solve() {
	int msg;
	msg=changeLayer(layerCoarsest);
	if(msg!=0) return msg;
	while(true) {
		msg=solveLayer();
		if(msg!=0) return msg;
		if(layer<layerFinest) {
			msg=refineDuals(layer+1);
			if(msg!=0) return msg;
			msg=changeLayer(layer+1);
			if(msg!=0) return msg;
		} else {
			if (cfg.generateFinalKernel) {
				eprintf("generating final kernel.\n");
				msg=generateKernel();
				return msg;
			} else {
				return 0;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// standard 2-marginal Sinkhorn algorithm
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TSinkhornSolverStandard::TSinkhornSolverStandard(
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
		) : TSinkhornSolverBase(_nLayers, _nEpsList, _epsLists, _layerCoarsest, _layerFinest, _errorGoal, _cfg),
		kernelGenerator(_HPX, _HPY,NULL, NULL,_costProvider,0) {
	
	HPX=_HPX;
	HPY=_HPY;
	muXH=_muXH;
	muYH=_muYH;
	rhoXH=_rhoXH;
	rhoYH=_rhoYH;
	alphaH=_alphaH;
	betaH=_betaH;
	costProvider=_costProvider;
	
	xres=0;
	yres=0;
	muX=NULL;
	muY=NULL;
	alpha=NULL;
	beta=NULL;


}


int TSinkhornSolverStandard::initialize() {
	// call basic init method
	TSinkhornSolverBase::initialize();
	
	kernelGenerator.changeLayer(rhoXH[layer], rhoYH[layer], layer);


	return 0;
}

int TSinkhornSolverStandard::refineDuals(const int newLayer) {
	// refine dual variables
	if(newLayer>0) {
		HPX->signal_refine_double(alphaH[newLayer-1],alphaH[newLayer],newLayer-1);
		HPY->signal_refine_double(betaH[newLayer-1],betaH[newLayer],newLayer-1);
	}
	kernelValid=false;
	return 0;
}

int TSinkhornSolverStandard::changeLayer(const int newLayer) {
	TSinkhornSolverBase::changeLayer(newLayer);
			
	// adjust cost function provider
	costProvider->setLayerBottom(layer);

	// adjust kernel generator
	kernelGenerator.changeLayer(rhoXH[layer],rhoYH[layer],layer);

	// set short cut variables for current layer
	xres=HPX->layers[layer]->nCells;
	yres=HPY->layers[layer]->nCells;
	muX=muXH[layer];
	muY=muYH[layer];
	alpha=alphaH[layer];
	beta=betaH[layer];

	// setup marginal variables
	u=TMarginalVector::Constant(xres,1.);
	v=TMarginalVector::Constant(yres,1.);
	
	
//	// map marginal measures and dual potentials
//	muX=Eigen::Map<TMarginalVector>(muXH[layer],xres);
//	muY=Eigen::Map<TMarginalVector>(muYH[layer],yres);
//	alpha=Eigen::Map<TMarginalVector>(alphaH[layer],xres);
//	beta=Eigen::Map<TMarginalVector>(betaH[layer],yres);

	
	return 0;

}


int TSinkhornSolverStandard::checkAbsorb(const double maxValue) {
	if((u.maxCoeff()>maxValue) || (v.maxCoeff()>maxValue)) {
		return MSG_ABSORB_REITERATE;
	}
	return 0;
}

int TSinkhornSolverStandard::absorb() {
	SinkhornAbsorbScaling(HPX, alphaH, u, layer, eps);
	SinkhornAbsorbScaling(HPY, betaH, v, layer, eps);
	return 0;	
}

int TSinkhornSolverStandard::generateKernel() {
	// set slack for effective cost based on thresh for kernel entry
	kernel=kernelGenerator.generate(eps,-eps*log(cfg.truncation_thresh));
	
	// alternative / experimental: directly set slack for effective cost, scaled with layer scale
//	eprintf("\t\tkernel thresh: %e\n",cfg.truncation_thresh*pow(2,2*(layerFinest-layer)));
//	kernel=kernelGenerator.generate(eps,cfg.truncation_thresh*pow(2,2*(layerFinest-layer)));

	kernelT=kernel.transpose();
	eprintf("\t\tkernel entries: %ld\n",kernel.nonZeros());
	return 0;
}


int TSinkhornSolverStandard::refineKernel() {
	kernel=kernelGenerator.refine(kernel,eps);
	kernelT=kernel.transpose();
	eprintf("\t\tkernel entries: (refinement) %ld\n",kernel.nonZeros());	
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Standard optimal transport model

int TSinkhornSolverStandard::iterate(const int n) {
	// standard sinkhorn iterations
	Eigen::Map<TMarginalVector> muXVec(muX,xres);
	Eigen::Map<TMarginalVector> muYVec(muY,yres);

	for(int i=0;i<n;i++) {
		u=muXVec.cwiseQuotient(kernel*v);
		v=muYVec.cwiseQuotient(kernelT*u);
	}
	if( (!u.allFinite()) || (!v.allFinite()) ) {
		return MSG_NANSCALING;
	}
	return 0;
}


int TSinkhornSolverStandard::getError(double * const result) {
	// return L1 error of first marginal
	TMarginalVector muXEff=u.cwiseProduct(kernel*v);
	if(muXEff.hasNaN()) {
		return MSG_NANINERROR;
	}
	Eigen::Map<TMarginalVector> muXVec(muX,xres);
	(*result)=(muXEff-muXVec).lpNorm<1>();
	return 0;

//	// return L1 error of second marginal
//	TMarginalVector muYEff=v.cwiseProduct(kernelT*u);
//	if(muYEff.hasNaN()) {
//		return MSG_NANINERROR;
//	}
//	Eigen::Map<TMarginalVector> muYVec(muY,yres);
//	(*result)=(muYEff-muYVec).lpNorm<1>();
//	return 0;


}

double TSinkhornSolverStandard::scoreTransportCost() {
	// evaluates linear transport cost term
	double result=0;
	
	// iterate over elements of kernel
	for (int x=0; x<kernel.outerSize(); x++) {
		for (TKernelMatrix::InnerIterator it(kernel,x); it; ++it) {
			int y=it.col();
			// cost contribution: cost * pi(x,y), where pi(x,y) = u(x)*v(y)*kernel(x,y)
			// (the latter is stored in iterator value)
			result+=costProvider->getCost(layer, x, y)*u[x]*v[y]*it.value();
		}
	}	
	return result;
}

double TSinkhornSolverStandard::scorePrimalUnreg() {
	// evalutes unregularized primal functional.
	// ignore marginal constraints, simply return transport cost term
	return scoreTransportCost();
}


TMarginalVector TSinkhornSolverStandard::getMarginalX() const {
	// compute 1st marginal of transport plan
	return u.cwiseProduct(kernel*v);
}


TMarginalVector TSinkhornSolverStandard::getMarginalY() const {
	// compute 2nd marginal of transport plan
	return v.cwiseProduct(kernelT*u);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// soft marginal constraints
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int TSinkhornSolverStandardLInf::getError(double * const result) {
	// return L1 error of first marginal
	TMarginalVector muXEff=u.cwiseProduct(kernel*v);
	if(muXEff.hasNaN()) {
		return MSG_NANINERROR;
	}
	Eigen::Map<TMarginalVector> muXVec(muX,xres);
	(*result)=(muXEff-muXVec).lpNorm<Eigen::Infinity>();
	return 0;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// soft marginal constraints
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// soft marginal class
TSoftMarginalFunction::~TSoftMarginalFunction() {}

//////////////////////////////////////////////////////////////////////////////////////////
// soft marginal class: KL
void TSoftMarginalFunctionKL::proxdiv(double *u, double *conv, double *mu, double *alpha, double kappa, double eps, int res) {
	for(int x=0;x<res;x++) {
		if(conv[x]>0) {
			u[x]=pow(mu[x]/conv[x],kappa/(kappa+eps))*exp(-alpha[x]/(kappa+eps));
		}
	}
}	

double TSoftMarginalFunctionKL::PDGapContribution(double *muEff, double *alpha, double *mu, double kappa, int res) {
	double result=0;
	for(int x=0;x<res;x++) {
		if(muEff[x]>0) {
			result+=kappa*(muEff[x]*log(muEff[x]/mu[x])-muEff[x]+exp(-alpha[x]/kappa)*mu[x]);
		}
	}
	return result;
}

double TSoftMarginalFunctionKL::f(const double * const muEff, const double * const mu, const double kappa, const int res) const {
	// computes KL divergence of muEff w.r.t. mu
	// muThresh: mu is assumed to be lower bounded by muThresh,
		// entries that are two small are replaced by muThresh
		// this is supposed to regularize KL a bit around the singularity around mu=0
		// to stabilize primal-dual-gap estimates
	
	return TSinkhornSolverKLMarginals::KL(muEff,mu,kappa,res,muThresh);
}


double TSoftMarginalFunctionKL::fDual(double *alpha, double *mu, double kappa, int res) {
	// computes dual of KL divergence of alpha w.r.t. mu
	return TSinkhornSolverKLMarginals::KLDual(alpha,mu,kappa,res);
}

//////////////////////////////////////////////////////////////////////////////////////////
// soft marginal class: L1
void TSoftMarginalFunctionL1::proxdiv(double *u, double *conv, double *mu, double *alpha, double kappa, double eps, int res) {
	for(int x=0;x<res;x++) {
		if(conv[x]>0) {
			u[x]=exp(std::min((kappa-alpha[x])/eps,std::max((-kappa-alpha[x])/eps,log(mu[x]/conv[x]))));
		}
	}
}	

double TSoftMarginalFunctionL1::PDGapContribution(double *muEff, double *alpha, double *mu, double kappa, int res) {
	double result=0;
	for(int x=0;x<res;x++) {
		result+=kappa*abs(muEff[x]-mu[x])+std::max(-kappa,-alpha[x])*mu[x];
	}
	return result;
}


double TSoftMarginalFunctionL1::f(const double * const muEff, const double * const mu, const double kappa, const int res) const {
	// ignore the +infty penalty for muEff<0, since this will never happen anyways, ignoring this helps avoiding numerical instabilities at zero
	double result=0;
	for(int x=0;x<res;x++) {
		result+=kappa*abs(muEff[x]-mu[x]);
	}
	return result;
}


double TSoftMarginalFunctionL1::fDual(double *alpha, double *mu, double kappa, int res) {
	// ignore +infty penalty for alpha>lambda, as this does not arise during iterations, ignoring this helps avoiding numerical instabilities at lambda
	double result=0;
	for(int x=0;x<res;x++) {
		result+=std::max(-kappa,alpha[x])*mu[x];
	}
	return result;
}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
// generic soft marginal sinkhorn solver
TSinkhornSolverSoftMarginal::TSinkhornSolverSoftMarginal(
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
		) : TSinkhornSolverStandard(
			_nLayers, _nEpsList, _epsLists,
			_layerCoarsest, _layerFinest,
			_errorGoal,
			_cfg,
			_HPX, _HPY, _muXH, _muYH, _rhoXH, _rhoYH, _alphaH, _betaH,
			_costProvider)
		{
	
	FX=_FX;
	FY=_FY;
	kappaX=_kappaX;
	kappaY=_kappaY;
}	

int TSinkhornSolverSoftMarginal::iterate(const int n) {
	for(int i=0;i<n;i++) {
		// update v
		TMarginalVector conv=kernelT*u;
		FY->proxdiv(v.data(),conv.data(),muY,beta,kappaY,eps,yres);
		// update u
		conv=kernel*v;
		FX->proxdiv(u.data(),conv.data(),muX,alpha,kappaX,eps,xres);
	}
	if(u.hasNaN() || v.hasNaN()) {
		return MSG_NANSCALING;
	}
	return 0;
}

int TSinkhornSolverSoftMarginal::getError(double * const result) {
	
	// prepare objects needed for evaluation of PD gap
	TMarginalVector alphaEff=Eigen::Map<TMarginalVector>(alpha,xres);
	for(int x=0;x<xres;x++) {
		alphaEff[x]+=eps*log(u[x]);
	}
	
	TMarginalVector betaEff=Eigen::Map<TMarginalVector>(beta,yres);
	for(int y=0;y<yres;y++) {
		betaEff[y]+=eps*log(v[y]);
	}
	
	TMarginalVector muXEff=u.cwiseProduct(kernel*v);
	TMarginalVector muYEff=v.cwiseProduct(kernelT*u);
	if(muXEff.hasNaN()) return MSG_NANINERROR;
	if(muYEff.hasNaN()) return MSG_NANINERROR;

	
	// add up different contributions to PD gap
	*result=0;
	
	// kernel part
	*result+=(muXEff.dot(alphaEff)+muYEff.dot(betaEff));

	// marginal X part
	*result+=FX->PDGapContribution(muXEff.data(), alphaEff.data(), muX, kappaX, xres);
	
	// marginal Y part
	*result+=FY->PDGapContribution(muYEff.data(), betaEff.data(), muY, kappaY, yres);
	
	return 0;

}

double TSinkhornSolverSoftMarginal::scorePrimalUnreg() {
	// evalutes unregularized primal functional
	double result=0;
	
	TMarginalVector muXEff=u.cwiseProduct(kernel*v);
	TMarginalVector muYEff=v.cwiseProduct(kernelT*u);

	// marginal terms
	result+=FX->f(muXEff.data(),muX,kappaX,xres);
	result+=FY->f(muYEff.data(),muY,kappaY,yres);
	
	// transport cost term
	result+=scoreTransportCost();
	
	return result;
}



void TSinkhornSolverSoftMarginal::setKappa(double _kappa) {
	kappaX=_kappa;
	kappaY=_kappa;
}

void TSinkhornSolverSoftMarginal::setKappa(double _kappaX, double _kappaY) {
	kappaX=_kappaX;
	kappaY=_kappaY;
}

//////////////////////////////////////////////////////////////////////////////////////////
// adapted version for kl soft marginal
TSinkhornSolverKLMarginals::TSinkhornSolverKLMarginals(
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
		) : TSinkhornSolverSoftMarginal(
			_nLayers, _nEpsList, _epsLists,
			_layerCoarsest, _layerFinest,
			_errorGoal,
			_cfg,
			_HPX, _HPY, _muXH, _muYH, _rhoXH, _rhoYH, _alphaH, _betaH,		
			_costProvider,NULL,NULL,_kappa,_kappa)
		{

	FX=new TSoftMarginalFunctionKL();
	FY=FX;
}	

TSinkhornSolverKLMarginals::~TSinkhornSolverKLMarginals() {
	delete FX;
}


double TSinkhornSolverKLMarginals::KL(const double * const muEff, const double * const mu, const double kappa, const int res, const double muThresh) {
	// computes KL divergence of muEff w.r.t. mu
	// muThresh: mu is assumed to be lower bounded by muThresh,
		// entries that are two small are replaced by muThresh
		// this is supposed to regularize KL a bit around the singularity around mu=0
		// to stabilize primal-dual-gap estimates
	
	double result=0;
	double muReg;
	
	for(int i=0;i<res;i++) {
		muReg=std::max(muThresh,mu[i]);
		
		if((muEff[i]>0) && (muReg>0)) {
			result+=muEff[i]*log(muEff[i]/muReg)-muEff[i]+muReg;
		}
		if((muEff[i]>0) && (muReg==0)) {
			return TSinkhornSolverSoftMarginal::DBL_INFINITY;
		}
		if((muEff[i]==0) && (muReg>0)) {
			result+=muReg;
		}
	}
	return kappa*result;
}


double TSinkhornSolverKLMarginals::KLDual(double *alpha, double *mu, double kappa, int res) {
	// computes dual of KL divergence of alpha w.r.t. mu
	
	double result=0;
	
	for(int i=0;i<res;i++) {
		result+=(exp(alpha[i]/kappa)-1)*mu[i];
	}
	return kappa*result;
}

