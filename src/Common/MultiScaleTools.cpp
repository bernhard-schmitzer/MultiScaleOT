#include"MultiScaleTools.h"


//////////////////////////////////////////
// NN Search
//////////////////////////////////////////


std::vector<int> THierarchicalNN::find(double *posX, double **posYH, double **radiiY,
		THierarchicalPartition *HPY, int layerBottom, int nElements) {

	int dim=HPY->dim;

	std::vector<int> result(0);

	int foundElements=0;
	// initialize search list with only one element at coarsest level
	TCandidate candidate;
	candidate.layer=0;
	candidate.z=0;
	candidate.dist=0.;
	
	TCandidateList candidateList(&candidate,1);

	// list where to collect refinement lists
	std::vector<TCandidate> data(0);
	
	// now run recursive refinement strategy
	while((candidateList.data.begin() != candidateList.data.end())) {
		// extract smallest element from front of list
		TCandidate smallestEntry=candidateList.data.front();
		// afterwards, remove element from list
		candidateList.data.pop_front();

		//eprintf("%d\t%d\t%e\n",smallestEntry.layer,smallestEntry.z,smallestEntry.dist);

		if(smallestEntry.layer<layerBottom) {
			//eprintf("refining\n");
			// refinement required
			int newLayer=smallestEntry.layer+1; // new layer

			// number of children of candidate on new layer
			int nChildren=HPY->layers[smallestEntry.layer]->nChildren[smallestEntry.z];
			// pointer to children
			int *children=HPY->layers[smallestEntry.layer]->children[smallestEntry.z];

			// for children obtain new entries
			data.resize(nChildren);			
			for(int i=0;i<nChildren;i++) {
				data[i].layer=newLayer;
				data[i].z=children[i];
				data[i].dist=std::pow(EUCL_lincombSqr(posX, posYH[newLayer]+dim*data[i].z, 1., -1., dim),0.5);
				if(newLayer<layerBottom) {
					data[i].dist-=radiiY[newLayer][data[i].z];
				}
			}
			
			// merge these new entries into list at appropriate locations
			candidateList.merge(data.data(),nChildren);			
			
		} else {
			//eprintf("found finest element\n");
			// front element of candidate list is now on finest level
			// therefore, add to result
			result.push_back(smallestEntry.z);
			foundElements++;
			// return list, when sufficient number of elements is found
			if(foundElements>=nElements) {
				return result;
			}
			
		}
	}
	// should the number of element never exceed nElements, simply return full list now
	return result;
}

TVarListHandler* THierarchicalNN::getNeighbours(double **posXH, double **radiiX,
		THierarchicalPartition *HPX, int layerBottom, int nElements) {

	int xres=HPX->layers[layerBottom]->nCells;
	int dim=HPX->dim;
	
	TVarListHandler *result=new TVarListHandler;
	result->setupEmpty(xres);
	
	for(int x=0;x<xres;x++) {
		// find nElements+1 nearest neighbours of point
		// +1 because nearest neighbour will always be point itself
		std::vector<int> line=find(posXH[layerBottom]+dim*x, posXH, radiiX,
				HPX, layerBottom, nElements+1);
		
		// add elements from 2nd onwards to result
		result->addToLine(x,line.data()+1,line.size()-1);
	}
	
	return result;
}

TVarListHandler** THierarchicalNN::getNeighboursH(double **posXH, double **radiiX,
		THierarchicalPartition *HPX, int nElements) {
	
	int nLayers=HPX->nLayers;
	TVarListHandler **result=(TVarListHandler**) malloc(sizeof(TVarListHandler*)*nLayers);
	for(int layer=0;layer<nLayers;layer++) {
		result[layer]=getNeighbours(posXH, radiiX,
				HPX, layer, nElements);
	}
	return result;
}




void THierarchicalDualMaximizer::getMaxDual(THierarchicalPartition *partitionX, THierarchicalPartition *partitionY,
		double **alpha, double **beta, int layerFine,
		THierarchicalCostFunctionProvider *costProvider,
		int mode) {

	// for one given dual variable, compute the maximal other one, using hierarchical search

	// mode=MODE_ALPHA: compute alpha for given beta
	// else: compute beta for given alpha

	THierarchicalPartition *HPA,*HPB;

	// deciding whether to fix row or col:
	// set HPA to primary partition
	// set HPB to secondary partition (e.g. HPY if row mode)
	if (mode==MODE_ALPHA) {
		HPA=partitionX;
		HPB=partitionY;
	} else {
		HPA=partitionY;
		HPB=partitionX;
	}
	// for queries to costProvider use simple if

	int ares; // number of entries in row or column that need to be computed
	ares=HPA->layers[layerFine]->nCells;

	for (int a=0; a<ares; a++) {

	
		// generate hierarchical row at coarsest level
		int nB=HPB->layers[0]->nCells;
		std::vector<THierarchicalDualMaximizer::TCandidate> data(nB);
		for(int iB=0;iB<nB;iB++) {
			data[iB].layer=0;
			data[iB].z=iB;
			if(mode==MODE_ALPHA) {
				data[iB].v=costProvider->getCostAsym(layerFine, a, 0, iB)-beta[0][iB];
			} else {
				data[iB].v=costProvider->getCostAsym(0, iB, layerFine, a)-alpha[0][iB];
			}
		}
	
		// create initial kernel row
		THierarchicalDualMaximizer::TCandidateList row(data.data(),nB);
	
		// now run recursive refinement strategy
		while((row.data.begin() != row.data.end())) {
			// extract smallest element from front of list
			THierarchicalDualMaximizer::TCandidate smallestEntry=row.data.front();
			// afterwards, remove element from list
			row.data.pop_front();

			if(smallestEntry.layer<layerFine) {
				// refinement required
				int newLayer=smallestEntry.layer+1; // new layer
				int nChildrenB=HPB->layers[smallestEntry.layer]->nChildren[smallestEntry.z]; // number of children of b on new layer
				int *childrenB=HPB->layers[smallestEntry.layer]->children[smallestEntry.z]; // pointer to b children

				// for children of b obtain new entries for hierarchical row
				data.resize(nChildrenB);			
				for(int iB=0;iB<nChildrenB;iB++) {
					data[iB].layer=newLayer;
					data[iB].z=childrenB[iB];
					if(mode==MODE_ALPHA) {
						data[iB].v=costProvider->getCostAsym(layerFine, a, newLayer, childrenB[iB])-beta[newLayer][childrenB[iB]];
					} else {
						data[iB].v=costProvider->getCostAsym(newLayer, childrenB[iB], layerFine, a)-alpha[newLayer][childrenB[iB]];
					}

				}
			
				// merge these new entries into list at appropriate locations
				row.merge(data.data(),nChildrenB);			
			
			} else {
				// front element of hiearchical row is now on finest level
				if(mode==MODE_ALPHA) {
					alpha[layerFine][a]=smallestEntry.v;
				} else {
					beta[layerFine][a]=smallestEntry.v;
				}
				break;
							
			}
		}
	}	

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TMultiScaleSetup
TMultiScaleSetup::TMultiScaleSetup(double *_pos, double *_mu, int _res, int _dim, int _depth, int _childMode,
		bool _setup, bool _setupDuals, bool _setupRadii
		) {
	pos=_pos;
	mu=_mu;
	res=_res;
	dim=_dim;
	depth=_depth;
	HierarchyBuilderChildMode=_childMode;

	posH=NULL;
	muH=NULL;
	resH=NULL;
	
	HB=NULL;
	HP=NULL;
	
	alphaH=NULL;
	radii=NULL;
	neighboursH=NULL;
	

	if(_setup) {
		Setup();
		if(_setupDuals) {
			SetupDuals();
		}
		if(_setupRadii) {
			SetupRadii();
		}
	}
}

TMultiScaleSetup::TMultiScaleSetup(THierarchyBuilder *_HB, double **_posH, double *_mu, int _res, int _dim,
		bool _setupDuals, bool _setupRadii
		) {
	HB=_HB;
	depth=HB->layers.size()-1;
	nLayers=HB->layers.size();
	pos=_posH[depth];
	mu=_mu;
	res=_res;
	dim=_dim;
	HierarchyBuilderChildMode=THierarchyBuilder::CM_Manual;
	
	HB=_HB;
	
	alphaH=NULL;
	radii=NULL;
	neighboursH=NULL;
	
	eprintf("\tconverting\n");
	// convert to format used by hierarchical solver
	HP=HB->convert();
	eprintf("\tresH\n");
	// get hierarchical cardinalities of each marginal on each level
	resH=HB->getResH();
	
	// hierarchical positions
	// position of nodes in coarser hierarchy levels
	eprintf("\tposH\n");
	posH=HB->allocateDoubleSignal(dim);
	for(int i=0;i<depth+1;i++) {
		std::memcpy(posH[i],_posH[i],sizeof(double)*resH[i]*dim);
	}
	
	// hierarchical masses
	// combined masses of nodes in coarser hierarchy levels
	eprintf("\tmuH\n");
	muH=HB->allocateDoubleSignal(1);
	HP->computeHierarchicalMasses(mu,muH);

	eprintf("\trest of setup\n");

	eprintf("\tduals\n");
	if(_setupDuals) {
		SetupDuals();
	}
	eprintf("\tradii\n");
	if(_setupRadii) {
		radii=HB->allocateDoubleSignal(1);
		HB->getSignalRadiiExplicit(posH,radii);
	}
}


TMultiScaleSetup::TMultiScaleSetup(TMultiScaleSetup&& b) {
	pos=b.pos;
	mu=b.mu;
	res=b.res;
	dim=b.dim;
	
	depth=b.depth;
	nLayers=b.nLayers;
	HB=b.HB;
	b.HB=NULL;
	HP=b.HP;
	b.HP=NULL;
	posH=b.posH;
	b.posH=NULL;
	muH=b.muH;
	b.muH=NULL;
	resH=b.resH;
	b.resH=NULL;
	HierarchyBuilderChildMode=b.HierarchyBuilderChildMode;
	alphaH=b.alphaH;
	b.alphaH=NULL;
	radii=b.radii;
	b.radii=NULL;
	neighboursH=b.neighboursH;
	b.neighboursH=NULL;

}


TMultiScaleSetup::~TMultiScaleSetup() {
	// free dual variables if allocated
	if(alphaH!=NULL) {
		HP->signal_free_double(alphaH, 0, HP->nLayers-1);
	}

	// free radii if allocated
	if(radii!=NULL) {
		HP->signal_free_double(radii, 0, HP->nLayers-2);
	}

	// free neighbours if assigned
	if(neighboursH!=NULL) {
		for(int i=0;i<nLayers;i++) {
			delete neighboursH[i];
		}
		free(neighboursH);
	}


	// free hiearchical partitions

	if(HB!=NULL) {

		free(resH);		
		HB->freeSignal(muH,HB->layers.size());
		HB->freeSignal(posH,HB->layers.size());
		delete HP;
		delete HB;
	}
	
}


int TMultiScaleSetup::BasicMeasureChecks() {

	// sanity check: mu must be strictly positive
	if(doubleArrayMin(mu,res)<=0.) {
		eprintf("ERROR: minimum of mu is not strictly positive.\n");
		return ERR_PREP_INIT_MUXNEG;
	}

	// TODO: more sanity checks?

	return 0;
}


int TMultiScaleSetup::SetupHierarchicalPartition() {
	
	// create hierarchical partition
	HB = new THierarchyBuilder(pos,res,dim, HierarchyBuilderChildMode, depth);
		
	// convert to format used by hierarchical solver
	HP=HB->convert();
	
	// hierarchical positions
	// position of nodes in coarser hierarchy levels
	posH=HB->allocateDoubleSignal(dim);
	HB->getSignalPos(posH);
	
	// hierarchical masses
	// combined masses of nodes in coarser hierarchy levels
	muH=HB->allocateDoubleSignal(1);
	HP->computeHierarchicalMasses(mu,muH);

	// get hierarchical cardinalities of each marginal on each level
	resH=HB->getResH();

	nLayers=HP->nLayers;
	
	return 0;
	
}

int TMultiScaleSetup::Setup() {
	int msg;

	msg=BasicMeasureChecks();
	if(msg!=0) { return msg; }
	msg=SetupHierarchicalPartition();
	if(msg!=0) { return msg; }	

	// only invoke these on demand
//	msg=SetupDuals();
//	if(msg!=0) { return msg; }	
//	msg=SetupRadii();
//	if(msg!=0) { return msg; }	


	return 0;

}


int TMultiScaleSetup::SetupDuals() {
	alphaH=HP->signal_allocate_double(0,HP->nLayers-1);
	return 0;
}

int TMultiScaleSetup::SetupRadii() {
	radii=HB->getSignalRadii();	
	return 0;
}


int TMultiScaleSetup::UpdatePositions(double *newPos) {
	pos=newPos;
	HB->updatePositions(pos);
	HB->getSignalPosExplicit(posH);
	
	if (radii!=NULL) {
		HB->getSignalRadiiExplicit(posH,radii);
	}
	
	return 0;
}

int TMultiScaleSetup::UpdateMeasure(double *newMu) {
	mu=newMu;
	if(muH!=NULL) {
		HP->computeHierarchicalMasses(mu,muH);	
	}
	return 0;
}


//////////////////////////////////////////
// Cartesian Grid
//////////////////////////////////////////



TMultiScaleSetupGrid::TMultiScaleSetupGrid(TDoubleMatrix *_muGrid, int _depth,
		bool _setup, bool _setupDuals, bool _setupRadii) :
		TMultiScaleSetup(NULL, _muGrid->data, 0, 0, _depth, THierarchyBuilder::CM_Grid, true, true, true),
		posExplicit(GridToolsGetGridMatrix(_muGrid->depth,_muGrid->dimensions)) {
	
	muGrid=_muGrid;
	ownMuGrid=false;
	
	pos=posExplicit.data;
	res=posExplicit.dimensions[0];
	dim=posExplicit.dimensions[1];
	dimH=NULL;


	if(_setup) {
		Setup();
		if(_setupDuals) {
			SetupDuals();
		}
		if(_setupRadii) {
			SetupRadii();
		}
	}
		
}

TMultiScaleSetupGrid::TMultiScaleSetupGrid(TMultiScaleSetupGrid&& b) :
		TMultiScaleSetup(std::move(b)), posExplicit(std::move(b.posExplicit)) {

	muGrid=b.muGrid;
	b.muGrid=NULL;
	ownMuGrid=b.ownMuGrid;
	b.ownMuGrid=false;
	dimH=b.dimH;
	b.dimH=NULL;



}

int TMultiScaleSetupGrid::SetupHierarchicalPartition() {
	int msg;
	
	msg=TMultiScaleSetup::SetupHierarchicalPartition(); // call original method
	if(msg!=0) {
		return msg;
	}
	
	// initialize dimH
	dimH=HB->getDimH(muGrid->dimensions);
	return 0;
}

int TMultiScaleSetupGrid::SetupGridNeighbours() {
	if(neighboursH!=NULL) {
		for(int i=0;i<nLayers;i++) {
			delete neighboursH[i];
		}
		free(neighboursH);
	}

	neighboursH=(TVarListHandler**) malloc(sizeof(TVarListHandler*)*nLayers);
	for(int i=0;i<nLayers;i++) {
		neighboursH[i]=new TVarListHandler();
		neighboursH[i]->setupEmpty(resH[i]);
		GridToolsGetNeighbours(dim, dimH+i*dim, neighboursH[i]);
	}
	
	return 0;
}


TMultiScaleSetupGrid::~TMultiScaleSetupGrid() {
	if(dimH!=NULL) {
		free(dimH);
	}
	if(ownMuGrid && (muGrid!=NULL)) {
		free(muGrid);
	}
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Barycenter

TMultiScaleSetupBarycenterContainer::TMultiScaleSetupBarycenterContainer() {
	nMarginals=0;
	HP=NULL;
	HPZ=NULL;
	muH=NULL;
	muZH=NULL;
	alphaH=NULL;
	betaH=NULL;
	costProvider=NULL;
	weights=NULL;
	resH=NULL;
	resZH=NULL;
}

TMultiScaleSetupBarycenterContainer::TMultiScaleSetupBarycenterContainer(const int _nMarginals)
	: TMultiScaleSetupBarycenterContainer() {
	setupEmpty(_nMarginals);
}

TMultiScaleSetupBarycenterContainer::~TMultiScaleSetupBarycenterContainer() {
	cleanup();
}

void TMultiScaleSetupBarycenterContainer::setupEmpty(const int _nMarginals) {
	cleanup();
	nMarginals=_nMarginals;
	HP=(THierarchicalPartition**) malloc(sizeof(THierarchicalPartition*)*nMarginals);
	muH=(double***) malloc(sizeof(double**)*nMarginals);
	costProvider=(THierarchicalCostFunctionProvider**) malloc(sizeof(THierarchicalCostFunctionProvider*)*nMarginals);
	weights=(double*) malloc(sizeof(double)*nMarginals);
	resH=(int**) malloc(sizeof(int*)*nMarginals);

}

void TMultiScaleSetupBarycenterContainer::cleanup() {
	if(alphaH!=NULL) {
		for(int i=0;i<nMarginals;i++) {
			if(alphaH[i]!=NULL) {
				HP[i]->signal_free_double(alphaH[i], 0, HP[i]->nLayers-1);
			}
		}
		free(alphaH);
	}
	if(betaH!=NULL) {
		for(int i=0;i<nMarginals;i++) {
			if(betaH[i]!=NULL) {
				HPZ->signal_free_double(betaH[i], 0, HPZ->nLayers-1);
			}
		}
		free(betaH);
	}
	if(HP!=NULL) {
		free(HP);
	}
	if(muH!=NULL) {
		free(muH);
	}
	if(costProvider!=NULL) {
		free(costProvider);
	}
	if(weights!=NULL) {
		free(weights);
	}
	if(resH!=NULL) {
		free(resH);
	}
}

void TMultiScaleSetupBarycenterContainer::setMarginal(const int n, TMultiScaleSetup &multiScaleSetup,
		const double weight) {
	HP[n]=multiScaleSetup.HP;
	muH[n]=multiScaleSetup.muH;
	//alphaH[n]=multiScaleSetup.alphaH;
	resH[n]=multiScaleSetup.resH;
	weights[n]=weight;
}

void TMultiScaleSetupBarycenterContainer::setCenterMarginal(TMultiScaleSetup &multiScaleSetup) {
	HPZ=multiScaleSetup.HP;
	muZH=multiScaleSetup.muH;
	resZH=multiScaleSetup.resH;
}


void TMultiScaleSetupBarycenterContainer::setupDuals() {

	alphaH=(double***) malloc(sizeof(double**)*nMarginals);
	betaH=(double***) malloc(sizeof(double**)*nMarginals);

	for(int i=0;i<nMarginals;i++) {
		alphaH[i]=HP[i]->signal_allocate_double(0,HP[i]->nLayers-1);
		betaH[i]=HPZ->signal_allocate_double(0,HPZ->nLayers-1);
	}
}

void TMultiScaleSetupBarycenterContainer::setCostFunctionProvider(
		int n, THierarchicalCostFunctionProvider &costFunctionProvider) {
	costProvider[n]=&costFunctionProvider;
	costProvider[n]->alpha=alphaH[n];
	costProvider[n]->beta=betaH[n];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class THierarchicalSearchList<THierarchicalNN::TCandidate>;

