#ifndef TEPSSCALING_H_
#define TEPSSCALING_H_

#include<cstdlib>
#include<vector>
#include<cmath>
#include<cstring>

#include<Common/ErrorCodes.h>
#include<Common/Verbose.h>


class TEpsScalingHandler {
public:
	// eps scaling over multiple layers
	int nLayers; // number of layers
	double **epsLists; // one eps list per layer
	int *nEpsLists; // number of eps in each layer	
	
	TEpsScalingHandler();
	~TEpsScalingHandler();
	

	int setupGeometricMultiLayerA(int _nLayers, double epsStart, double epsTarget, int epsSteps, double boxScale, double layerExponent, int layerCoarsest, bool overlap);
	int setupGeometricMultiLayerB(int _nLayers, double epsBase, double layerFactor, int layerSteps, int stepsFinal);
	int setupGeometricSingleLayer(int _nLayers, double epsStart, double epsTarget, int epsSteps);
	int setupExplicit(int _nLayers, double **_epsLists, int *_nEpsLists);

	void initLayers(int _nLayers);
	double* getGeometricEpsList(double epsStart, double epsTarget, int epsSteps);
	double* getEpsScalesFromBox(double boxScale, double layerExponent);
	int getEpsScalingSplit(double *epsList, double *epsScales, int nEps, int levelCoarsest, bool overlap);
};

#endif
