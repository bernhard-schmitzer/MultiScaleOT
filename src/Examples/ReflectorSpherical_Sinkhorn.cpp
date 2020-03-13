#include<cstdlib>
#include <vector>

#include<Common.h>
#include<Sinkhorn.h>

int main() {
	#ifdef VERBOSE
	#ifdef VERBOSE_DYN
	verbose_mode=true;
	#endif
	#endif

	int msg;
	
	// read raw data from file
	char filenamePosX[]="data/reflector_sphere/posX_9771.dat";
	char filenamePosY[]="data/reflector_sphere/posY_10800_Monge.dat";
	char filenameMuX[]="data/reflector_sphere/muX_9771.dat";
	char filenameMuY[]="data/reflector_sphere/muY_10800_Monge.dat";

	std::vector<double> posXdat=readFile<double>(filenamePosX);
	std::vector<double> posYdat=readFile<double>(filenamePosY);
	std::vector<double> muXdat=readFile<double>(filenameMuX);
	std::vector<double> muYdat=readFile<double>(filenameMuY);

	int dim=3;
	int xres=muXdat.size();
	int yres=muYdat.size();


	///////////////////////////////////////////////
	int depth=6;
	int layerCoarsest=0;
	int layerFinest=depth;

	
	///////////////////////////////////////////////

	TMultiScaleSetupSphere MultiScaleSetupX(posXdat.data(),muXdat.data(),xres,dim,depth);
	TMultiScaleSetupSphere MultiScaleSetupY(posYdat.data(),muYdat.data(),yres,dim,depth);


	
	eprintf("hierarchical cardinalities:\n");
	for(int layer=0;layer<MultiScaleSetupX.nLayers;layer++) {
		eprintf("%d\t%d\n",layer,MultiScaleSetupX.HP->layers[layer]->nCells);
	}
	
	
	/////////////////////////////////////////////////////////////////////////////////////////
	// setup a cost function provider
	THierarchicalCostFunctionProvider_Reflector costProvider(
		MultiScaleSetupX.posH, MultiScaleSetupY.posH,
		MultiScaleSetupX.radii, MultiScaleSetupY.radii,
		dim, 0,
		true,
		MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH
		);
	
	
	/////////////////////////////////////////////////////////////////////////////////////////
	// epsScaling
	double epsStart=1E1;
	double epsTarget=2E-5;
	int epsSteps=25;
	double epsBoxScale=.7;
	
	TEpsScalingHandler epsScalingHandler;
	msg=epsScalingHandler.setupGeometricMultiLayerA(MultiScaleSetupX.nLayers,epsStart,epsTarget,epsSteps,epsBoxScale,2.,layerCoarsest,true);
	if(msg!=0) { eprintf("error: %d\n",msg); return msg; }

	/////////////////////////////////////////////////////////////////////////////////////////
	// errorGoal
	double errorGoal=1E-5;
	// other parameters
	TSinkhornSolverBase::TSinkhornSolverParameters cfg=TSinkhornSolverBase::SinkhornSolverParametersDefault;
	cfg.truncation_thresh=1E-6;
	cfg.refineKernel=true;

	/////////////////////////////////////////////////////////////////////////////////////////
	// create solver object
	TSinkhornSolverStandard SinkhornSolver(MultiScaleSetupX.nLayers, epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
			layerCoarsest, layerFinest,
			errorGoal,
			cfg,
			MultiScaleSetupX.HP, MultiScaleSetupY.HP,
			MultiScaleSetupX.muH, MultiScaleSetupY.muH,
			MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH,
			&costProvider
			);
	

	SinkhornSolver.initialize();
	msg=SinkhornSolver.solve();	
	printf("return code: %d\n",msg);

	
	// recompute kernel one last time
	SinkhornSolver.generateKernel();
	// return primal objective value
	double primalScore=SinkhornSolver.scorePrimalUnreg();
	printf("primal score: %e\n",primalScore);

	
	return 0;

}
