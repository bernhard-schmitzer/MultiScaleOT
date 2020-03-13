#include"TEpsScaling.h"


TEpsScalingHandler::TEpsScalingHandler() {
	nLayers=0;
	epsLists=NULL;
	nEpsLists=NULL;
}



int TEpsScalingHandler::setupGeometricMultiLayerA(int _nLayers, double epsStart, double epsTarget, int epsSteps, double boxScale, double layerExponent, int layerCoarsest, bool overlap) {
	initLayers(_nLayers);
	
	double *epsList=getGeometricEpsList(epsStart, epsTarget, epsSteps);


	double *epsScales=getEpsScalesFromBox(boxScale, layerExponent);
		
	int msg=getEpsScalingSplit(epsList, epsScales, epsSteps+1, layerCoarsest, overlap);
		
	free(epsScales);
	free(epsList);
	return msg;
}

int TEpsScalingHandler::setupGeometricMultiLayerB(int _nLayers, double epsBase, double layerFactor, int layerSteps, int stepsFinal) {
	initLayers(_nLayers);
	
	double stepFactor=pow(layerFactor,1./layerSteps);
	
	for(int l=0;l<nLayers;l++) {
		if(l<nLayers-1) {
			epsLists[l]=(double*) malloc(sizeof(double)*(layerSteps+1));
			nEpsLists[l]=layerSteps+1;
		} else {
			epsLists[l]=(double*) malloc(sizeof(double)*(layerSteps+1+stepsFinal));
			nEpsLists[l]=layerSteps+1+stepsFinal;
		}
		for(int i=0;i<layerSteps+1;i++) {
			epsLists[l][i]=epsBase*pow(layerFactor,nLayers-l-1)*pow(stepFactor,layerSteps-i);
		}
		if(l==nLayers-1) {
		for(int i=0;i<stepsFinal;i++) {
			epsLists[l][layerSteps+1+i]=epsBase*pow(stepFactor,-i-1);
		}
		}
	}
	return 0;
}


int TEpsScalingHandler::setupGeometricSingleLayer(int _nLayers, double epsStart, double epsTarget, int epsSteps) {
	initLayers(_nLayers);
	
	nEpsLists[_nLayers-1]=epsSteps+1;
	epsLists[_nLayers-1]=getGeometricEpsList(epsStart, epsTarget, epsSteps);

	return 0;
}


int TEpsScalingHandler::setupExplicit(int _nLayers, double **_epsLists, int *_nEpsLists) {
	initLayers(_nLayers);
	for(int i=0;i<_nLayers;i++) {
		nEpsLists[i]=_nEpsLists[i];
		epsLists[i]=(double*) malloc(sizeof(double)*nEpsLists[i]);
		memcpy(epsLists[i],_epsLists[i],sizeof(double)*nEpsLists[i]);
	}	
	return 0;
}


TEpsScalingHandler::~TEpsScalingHandler() {
	if(nEpsLists!=NULL) {
		for(int i=0;i<nLayers;i++) {
			if(epsLists[i]!=NULL) {
				free(epsLists[i]);
			}
		}
		free(epsLists);
		free(nEpsLists);
	}
}

void TEpsScalingHandler::initLayers(int _nLayers) {
	nLayers=_nLayers;
	nEpsLists=(int*) malloc(sizeof(int)*nLayers);
	epsLists=(double**) malloc(sizeof(double*)*nLayers);
	for(int i=0;i<nLayers;i++) {
		nEpsLists[i]=0;
		epsLists[i]=NULL;
	}
}


double* TEpsScalingHandler::getGeometricEpsList(double epsStart, double epsTarget, int epsSteps) {

	double *epsList=(double*) malloc(sizeof(double)*(epsSteps+1));
	
	epsList[0]=epsStart;
	if (epsSteps==0) {
		return epsList;
	}
	
	epsList[epsSteps]=epsTarget;
	
	for(int i=1;i<epsSteps;i++) {
		epsList[i]=pow(epsStart,1.-(double)i/epsSteps)*pow(epsTarget,(double)i/epsSteps);
	}
	return epsList;
}

double* TEpsScalingHandler::getEpsScalesFromBox(double boxScale, double layerExponent) {
	double *epsScales=(double*) malloc(sizeof(double)*nLayers);
	for(int i=0;i<nLayers-1;i++) {
		epsScales[i]=pow(boxScale*pow(0.5,i),layerExponent);
	}
	epsScales[nLayers-1]=0;
	return epsScales;
}



int TEpsScalingHandler::getEpsScalingSplit(double *epsList, double *epsScales, int nEps, int levelCoarsest, bool overlap) {


	std::vector<std::vector<double> > vecEpsLists(nLayers);

	// divide eps list over scales
	int currentScale=levelCoarsest;
	for(int i=0;i<nEps;i++) {
		double eps=epsList[i];
		while(eps<epsScales[currentScale]) {
			currentScale++;
		}
		vecEpsLists[currentScale].push_back(eps);
	}

	// create overlaps
	if(overlap) {
		for(int i=1;i<nLayers;i++) {
			if(vecEpsLists[i-1].size()>0) {
				// if prev eps list is not empty
				int nPrev=vecEpsLists[i-1].size();
				vecEpsLists[i].insert(vecEpsLists[i].begin(), vecEpsLists[i-1][nPrev-1]);
			}
		}
	}

	// check if coarsest eps list has at least one element
	if(vecEpsLists[levelCoarsest].size()==0) {
		return ERR_EPSSCALING_EMPTYCOARSESUBLIST;
	}
	
	// convert eps list to oldschool array format
	for(int i=0;i<nLayers;i++) {
		nEpsLists[i]=vecEpsLists[i].size();
		if(nEpsLists[i]>0) {
			epsLists[i]=(double*) malloc(sizeof(double)*(nEpsLists[i]));
			for(int j=0;j<nEpsLists[i];j++) {
				epsLists[i][j]=vecEpsLists[i][j];
			}
		} else {
			epsLists[i]=NULL;
		}
	}
	
	return 0;
	
}


