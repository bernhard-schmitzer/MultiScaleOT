#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Sinkhorn.h>

constexpr int EXAMPLE_MODE_STANDARD=0;
constexpr int EXAMPLE_MODE_UNBALANCED=1;


int main(int argc, char* argv[]);
int example_minimal(int example_mode, double HKscale);
int example_file(int example_mode, double HKscale);

int main(int argc, char* argv[]) {
	#ifdef VERBOSE
	#ifdef VERBOSE_DYN
	verbose_mode=true;
	#endif
	#endif

	char option;
	if(argc<=1) {
		option='0';
	} else {
		option=argv[1][0];
	}
	switch(option) {
		case '3':
			example_file(EXAMPLE_MODE_UNBALANCED,48.);
			break;
		case '2':
			example_file(EXAMPLE_MODE_STANDARD,0.);
			break;
		case '1':
			example_minimal(EXAMPLE_MODE_UNBALANCED,5.);
			break;
		default:
			example_minimal(EXAMPLE_MODE_STANDARD,0.);
			break;
	}			
}


int example_minimal(int example_mode, double HKscale) {

	// two reference measures, given as point clouds
	double muXdat[]={0.5, 0.5};
	double muYdat[]={0.5, 0.5};

	double posXdat[]={0., 1.};
	double posYdat[]={2., 3.};
	
		
	int posXdim[]={2, 1};
	int posYdim[]={2, 1};
	
	///////////////////////////////////////////////
	// setup problem data

	TDoubleMatrix posX,posY;

	posX.data=posXdat;
	posX.dimensions=posXdim;
	posX.depth=2;
	
	posY.data=posYdat;
	posY.dimensions=posYdim;
	posY.depth=2;
	
	///////////////////////////////////////////////
	// reference measure lists
	TDoubleMatrix* pos[2]={&posX,&posY};
	double* mu[2]={muXdat,muYdat};
	
	// barycenter weights for the reference measures
	double weights[2]={0.5,0.5};
	int nMarginals=2;
	
	// central space
	int centerDim=1;
	int centerDimensions[]={4};
	int centerNPoints=GridToolsGetTotalPoints(centerDim, centerDimensions);
	TDoubleMatrix posCenter=GridToolsGetGridMatrix(centerDim, centerDimensions);
	std::vector<double> muCenter(centerNPoints,1./centerNPoints);
	
	
	// fundamental parameters
	int depth=2;
	int layerCoarsest=1;
	int layerFinest=depth;	
	int dim=posX.dimensions[1]; // spatial dimension
	// manage unbalanced mode
	double HKKLweight=std::pow(HKscale,2);
	bool HKmode;
	double errorGoal;
	switch(example_mode) {
	case EXAMPLE_MODE_UNBALANCED:
		errorGoal=1E-4;
		HKmode=true;
		break;
	default:
		errorGoal=1E-4;
		HKmode=false;
		break;
	}

	/////////////////////////////////////////////////////////////////////////////////////////

	int msg; // store return codes from functions

	/////////////////////////////////////////////////////////////////////////////////////////
	// basic hierarchical setup

	TMultiScaleSetup** MultiScaleSetupList=(TMultiScaleSetup**) malloc(sizeof(TMultiScaleSetup*)*nMarginals);
	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupList[i]=new TMultiScaleSetup(pos[i]->data,mu[i],pos[i]->dimensions[0],pos[i]->dimensions[1],depth,THierarchyBuilder::CM_Tree,true,false,true);
	}
	

	TMultiScaleSetup MultiScaleSetupCenter(posCenter.data,muCenter.data(),posCenter.dimensions[0],posCenter.dimensions[1],depth,THierarchyBuilder::CM_Tree,true,false,true);


	/////////////////////////////////////////////////////////////////////////////////////////
	// setup a cost function providers
	// need one cost function provider between center and every reference measure
	
	// allocate list of pointers to each cost function provider
	THierarchicalCostFunctionProvider **costProvider=
		(THierarchicalCostFunctionProvider**) malloc(sizeof(THierarchicalCostFunctionProvider*)*nMarginals);
	
	for(int i=0;i<nMarginals;i++) {
		// instantiate cost function provider between i-th reference measure and center
		costProvider[i]=new THierarchicalCostFunctionProvider_SquaredEuclidean(
			MultiScaleSetupList[i]->posH, MultiScaleSetupCenter.posH,
			MultiScaleSetupList[i]->radii, MultiScaleSetupCenter.radii,
			dim, layerFinest,
			true,
			NULL, NULL,
			1.,
			HKmode, HKscale);
	}

	
	/////////////////////////////////////////////////////////////////////////////////////////
	// barycenter setup object
	TMultiScaleSetupBarycenterContainer *MultiScaleSetupBarycenter= new TMultiScaleSetupBarycenterContainer(nMarginals);
	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupBarycenter->setMarginal(i,*MultiScaleSetupList[i],weights[i]);
	}
	MultiScaleSetupBarycenter->setCenterMarginal(MultiScaleSetupCenter);
	MultiScaleSetupBarycenter->setupDuals();

	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupBarycenter->setCostFunctionProvider(i, *costProvider[i]);
	}
	/////////////////////////////////////////////////////////////////////////////////////////
	// epsScaling
	double epsStart=1E2;
	double epsTarget=1E-1;
	int epsSteps=15;
	double epsBoxScale=centerDimensions[0]*0.75;
	
	TEpsScalingHandler epsScalingHandler;
	msg=epsScalingHandler.setupGeometricMultiLayerA(MultiScaleSetupCenter.nLayers,epsStart,epsTarget,epsSteps,epsBoxScale,2.,layerCoarsest,true);
	if(msg!=0) { eprintf("error: %d\n",msg); return msg; }


	/////////////////////////////////////////////////////////////////////////////////////////
	// other parameters
	// other parameters
	TSinkhornSolverBase::TSinkhornSolverParameters cfg=TSinkhornSolverBase::SinkhornSolverParametersDefault;
	cfg.absorption_scalingBound=1E2;
	cfg.absorption_scalingLowerBound=1E2;
	cfg.truncation_thresh=1E-10;
	cfg.refineKernel=false;

	/////////////////////////////////////////////////////////////////////////////////////////
	// create solver object

	TSinkhornSolverBarycenter *SinkhornSolver;
	// hard marginals
	if(HKmode) {
		SinkhornSolver = new TSinkhornSolverBarycenterKLMarginals(MultiScaleSetupCenter.nLayers,
				epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
				layerCoarsest, layerFinest,
				errorGoal,
				cfg,
				nMarginals,
				weights,
				MultiScaleSetupBarycenter->HP, MultiScaleSetupBarycenter->HPZ,
				MultiScaleSetupBarycenter->muH, MultiScaleSetupBarycenter->muZH,
				MultiScaleSetupBarycenter->alphaH, MultiScaleSetupBarycenter->betaH,
				costProvider,
				HKKLweight
				);
	} else {
		SinkhornSolver = new TSinkhornSolverBarycenter(MultiScaleSetupCenter.nLayers,
				epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
				layerCoarsest, layerFinest,
				errorGoal,
				cfg,
				nMarginals,
				weights,
				MultiScaleSetupBarycenter->HP, MultiScaleSetupBarycenter->HPZ,
				MultiScaleSetupBarycenter->muH, MultiScaleSetupBarycenter->muZH,
				MultiScaleSetupBarycenter->alphaH, MultiScaleSetupBarycenter->betaH,
				costProvider
				);
	}


	/////////////////////////////////////////////////////////////////////////////////////////
	// initialize
	SinkhornSolver->initialize();
	// solve
	msg=SinkhornSolver->solve();	
	printf("return code: %d\n",msg);
	if(msg!=0) { printf("error: %d\n",msg); return msg; }

	

	// compute mean center marginal
	TMarginalVector barycenter=TMarginalVector::Constant(centerNPoints,0.);
	for(int i=0;i<nMarginals;i++) {
		barycenter+=SinkhornSolver->v[i].cwiseProduct(SinkhornSolver->kernelT[i]*SinkhornSolver->u[i]);
	}
	barycenter=barycenter/nMarginals;
	// print
	printf("\nbarycenter:\n");
	for(int i=0;i<centerNPoints;i++) {
		printf(" %e",barycenter[i]);
	}
	printf("\n\n");


	// clean up memory	
	delete SinkhornSolver;
	delete MultiScaleSetupBarycenter;
	for(int i=0;i<nMarginals;i++) {
		delete costProvider[i];
		delete MultiScaleSetupList[i];
	}
	free(costProvider);
	free(MultiScaleSetupList);
		
	return 0;

}


int example_file(int example_mode, double HKscale) {

	// read raw data from file
	char filenamePoints0[]="data/barycenter/simple/64_0_pos.dat";
	char filenamePoints1[]="data/barycenter/simple/64_1_pos.dat";
	char filenamePoints2[]="data/barycenter/simple/64_2_pos.dat";

	char filenameMu0[]="data/barycenter/simple/64_0_mu.dat";
	char filenameMu1[]="data/barycenter/simple/64_1_mu.dat";
	char filenameMu2[]="data/barycenter/simple/64_2_mu.dat";

//	char filenamePoints0[]="data/barycenter/simple/256_0_pos.dat";
//	char filenamePoints1[]="data/barycenter/simple/256_1_pos.dat";
//	char filenamePoints2[]="data/barycenter/simple/256_2_pos.dat";

//	char filenameMu0[]="data/barycenter/simple/256_0_mu.dat";
//	char filenameMu1[]="data/barycenter/simple/256_1_mu.dat";
//	char filenameMu2[]="data/barycenter/simple/256_2_mu.dat";

//	char filenamePoints0[]="data/barycenter/groups/256_0_pos.dat";
//	char filenamePoints1[]="data/barycenter/groups/256_1_pos.dat";
//	char filenamePoints2[]="data/barycenter/groups/256_2_pos.dat";

//	char filenameMu0[]="data/barycenter/groups/256_0_mu.dat";
//	char filenameMu1[]="data/barycenter/groups/256_1_mu.dat";
//	char filenameMu2[]="data/barycenter/groups/256_2_mu.dat";


	std::vector<double> posDat0=readFile<double>(filenamePoints0);
	std::vector<double> posDat1=readFile<double>(filenamePoints1);
	std::vector<double> posDat2=readFile<double>(filenamePoints2);
	std::vector<double> muDat0=readFile<double>(filenameMu0);
	std::vector<double> muDat1=readFile<double>(filenameMu1);
	std::vector<double> muDat2=readFile<double>(filenameMu2);


	///////////////////////////////////////////////
	// setup problem data

	int dim=2;
	TDoubleMatrix pos0,pos1,pos2;

	pos0.data=posDat0.data();
	int pos0dim[]={(int) muDat0.size(), dim};
	pos0.dimensions=pos0dim;
	pos0.depth=2;

	pos1.data=posDat1.data();
	int pos1dim[]={(int) muDat1.size(), dim};
	pos1.dimensions=pos1dim;
	pos1.depth=2;

	pos2.data=posDat2.data();
	int pos2dim[]={(int) muDat2.size(), dim};
	pos2.dimensions=pos2dim;
	pos2.depth=2;
	
	
	///////////////////////////////////////////////	
	// reference measure lists
	TDoubleMatrix* pos[3]={&pos0,&pos1,&pos2};
	double* mu[3]={muDat0.data(),muDat1.data(),muDat2.data()};
	
	// barycenter weights for the reference measures
	double weights[3]={0.25,0.25,0.5};
	int nMarginals=3;
	
	// central space
	int centerDim=2;
	//int centerDimensions[]={256,256};
	int centerDimensions[]={64,64};
	int centerNPoints=GridToolsGetTotalPoints(centerDim, centerDimensions);
	TDoubleMatrix posCenter=GridToolsGetGridMatrix(centerDim, centerDimensions);
	std::vector<double> muCenter(centerNPoints,1./centerNPoints);
	
	
	// fundamental parameters
	int depth=6;
	int layerCoarsest=2;
	int layerFinest=depth;
	// manage unbalanced mode
	double HKKLweight=std::pow(HKscale,2);
	bool HKmode;
	double errorGoal;
	switch(example_mode) {
	case EXAMPLE_MODE_UNBALANCED:
		errorGoal=1E-1;
		HKmode=true;
		break;
	default:
		errorGoal=1E-3;
		HKmode=false;
		break;
	}

	/////////////////////////////////////////////////////////////////////////////////////////

	int msg; // store return codes from functions

	/////////////////////////////////////////////////////////////////////////////////////////
	// basic hierarchical setup
	TMultiScaleSetup** MultiScaleSetupList=(TMultiScaleSetup**) malloc(sizeof(TMultiScaleSetup*)*nMarginals);
	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupList[i]=new TMultiScaleSetup(pos[i]->data,mu[i],pos[i]->dimensions[0],pos[i]->dimensions[1],depth,THierarchyBuilder::CM_Tree,true,false,true);
	}
	

	TMultiScaleSetup MultiScaleSetupCenter(posCenter.data,muCenter.data(),posCenter.dimensions[0],posCenter.dimensions[1],depth,THierarchyBuilder::CM_Tree,true,false,true);


	/////////////////////////////////////////////////////////////////////////////////////////
	// setup a cost function providers
	// need one cost function provider between center and every reference measure
	
	// allocate list of pointers to each cost function provider
	THierarchicalCostFunctionProvider **costProvider=
		(THierarchicalCostFunctionProvider**) malloc(sizeof(THierarchicalCostFunctionProvider*)*nMarginals);
	
	for(int i=0;i<nMarginals;i++) {
		// instantiate cost function provider between i-th reference measure and center
		costProvider[i]=new THierarchicalCostFunctionProvider_SquaredEuclidean(
			MultiScaleSetupList[i]->posH, MultiScaleSetupCenter.posH,
			MultiScaleSetupList[i]->radii, MultiScaleSetupCenter.radii,
			dim, layerFinest,
			true,
			NULL,NULL,
			1.,
			HKmode, HKscale);
	}
	/////////////////////////////////////////////////////////////////////////////////////////
	// barycenter setup object
	TMultiScaleSetupBarycenterContainer *MultiScaleSetupBarycenter = new TMultiScaleSetupBarycenterContainer(nMarginals);
	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupBarycenter->setMarginal(i,*MultiScaleSetupList[i],weights[i]);
	}
	MultiScaleSetupBarycenter->setCenterMarginal(MultiScaleSetupCenter);
	MultiScaleSetupBarycenter->setupDuals();

	for(int i=0;i<nMarginals;i++) {
		MultiScaleSetupBarycenter->setCostFunctionProvider(i, *costProvider[i]);
	}

	/////////////////////////////////////////////////////////////////////////////////////////
	// epsScaling
	double epsStart=1E4;
	double epsTarget=1E-1;
	int epsSteps=25;
	double epsBoxScale=centerDimensions[0]*1.;
	
	TEpsScalingHandler epsScalingHandler;
	msg=epsScalingHandler.setupGeometricMultiLayerA(MultiScaleSetupCenter.nLayers,epsStart,epsTarget,epsSteps,epsBoxScale,2.,layerCoarsest,true);
	if(msg!=0) { eprintf("error: %d\n",msg); return msg; }

	/////////////////////////////////////////////////////////////////////////////////////////
	// other parameters
	TSinkhornSolverBase::TSinkhornSolverParameters cfg=TSinkhornSolverBase::SinkhornSolverParametersDefault;
	cfg.absorption_scalingBound=1E2;
	cfg.absorption_scalingLowerBound=1E2;
	cfg.truncation_thresh=1E-10;
	cfg.refineKernel=true;


	/////////////////////////////////////////////////////////////////////////////////////////
	// create solver object

	TSinkhornSolverBarycenter *SinkhornSolver;
	// hard marginals
	if(HKmode) {
		SinkhornSolver = new TSinkhornSolverBarycenterKLMarginals(MultiScaleSetupCenter.nLayers,
				epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
				layerCoarsest, layerFinest,
				errorGoal,
				cfg,
				nMarginals,
				weights,
				MultiScaleSetupBarycenter->HP, MultiScaleSetupBarycenter->HPZ,
				MultiScaleSetupBarycenter->muH, MultiScaleSetupBarycenter->muZH,
				MultiScaleSetupBarycenter->alphaH, MultiScaleSetupBarycenter->betaH,
				costProvider,
				HKKLweight
				);
	} else {
		SinkhornSolver = new TSinkhornSolverBarycenter(MultiScaleSetupCenter.nLayers,
				epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
				layerCoarsest, layerFinest,
				errorGoal,
				cfg,
				nMarginals,
				weights,
				MultiScaleSetupBarycenter->HP, MultiScaleSetupBarycenter->HPZ,
				MultiScaleSetupBarycenter->muH, MultiScaleSetupBarycenter->muZH,
				MultiScaleSetupBarycenter->alphaH, MultiScaleSetupBarycenter->betaH,
				costProvider
				);
	}


	/////////////////////////////////////////////////////////////////////////////////////////
	// initialize
	SinkhornSolver->initialize();
	// solve
	msg=SinkhornSolver->solve();	
	printf("return code: %d\n",msg);
	if(msg!=0) { printf("error: %d\n",msg); return msg; }

	

	// compute mean center marginal
	TMarginalVector barycenter=TMarginalVector::Constant(centerNPoints,0.);
	for(int i=0;i<nMarginals;i++) {
		barycenter+=SinkhornSolver->v[i].cwiseProduct(SinkhornSolver->kernelT[i]*SinkhornSolver->u[i]);
	}
	barycenter=barycenter/nMarginals;
	// write to file
	if(HKmode) {
		//writeFile<double>("results/barycenter_simple-64_unbalanced.dat", barycenter.data(), centerNPoints);
		writeFile<double>("results/barycenter_groups-256_unbalanced.dat", barycenter.data(), centerNPoints);
	} else {
		//writeFile<double>("results/barycenter_simple-64.dat", barycenter.data(), centerNPoints);
		writeFile<double>("results/barycenter_groups-256.dat", barycenter.data(), centerNPoints);
	}


	// clean up memory	
	delete SinkhornSolver;
	delete MultiScaleSetupBarycenter;
	for(int i=0;i<nMarginals;i++) {
		delete costProvider[i];
		delete MultiScaleSetupList[i];
	}
	free(costProvider);
	free(MultiScaleSetupList);
		
	return 0;

}

