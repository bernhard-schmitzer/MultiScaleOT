#include"THierarchicalPartition.h"

const int THierarchicalPartition::MODE_MIN=0;
const int THierarchicalPartition::MODE_MAX=1;
const int THierarchicalPartition::INTERPOLATION_MODE_CONSTANT=0;
const int THierarchicalPartition::INTERPOLATION_MODE_GRIDLINEAR=1;


TPartitionLayer::TPartitionLayer() {
	nCells=0;
	parent=NULL;
	children=NULL;
	leaves=NULL;
	nChildren=NULL;
	nLeaves=NULL;
}

TPartitionLayer::~TPartitionLayer() {

	// parent
	if(parent!=NULL) {
		free(parent);
	}

	// children
	if (children!=NULL) {
		for(int i=0;i<nCells;i++) {
			if(children[i]!=NULL) {
				free(children[i]);
			}
		}
		free(children);
	}
	
	// leaves
	if (leaves!=NULL) {
		for(int i=0;i<nCells;i++) {
			if(leaves[i]!=NULL) {
				free(leaves[i]);
			}
		}
		free(leaves);
	}
	
	// nChildren
	if (nChildren!=NULL) {
		free(nChildren);
	}
	
	// nLeaves
	if (nLeaves!=NULL) {
		free(nLeaves);
	}
}

void TPartitionLayer::initializeEmpty(int _nCells) {
	nCells=_nCells;
	// allocate space for children and leave lists
	
	children=(int**) malloc(sizeof(int*)*nCells);
	leaves=(int**) malloc(sizeof(int*)*nCells);

	nChildren=(int*) malloc(sizeof(int)*nCells);
	nLeaves=(int*) malloc(sizeof(int)*nCells);

	for(int i=0;i<nCells;i++) {
		children[i]=NULL;
		leaves[i]=NULL;
		nChildren[i]=0;
		nLeaves[i]=0;
	}
}


THierarchicalPartition::THierarchicalPartition(int _nLayers, int _dim, int _interpolation_mode) {
	nLayers=_nLayers;
	dim=_dim;
	interpolation_mode=_interpolation_mode;
	
	layers = (TPartitionLayer**) malloc(sizeof(TPartitionLayer*)*nLayers);
	for (int i=0;i<nLayers;i++) {
		layers[i]=NULL;
	}
}

THierarchicalPartition::~THierarchicalPartition() {
	for(int i=0;i<nLayers;i++) {
		if(layers[i]!=NULL) {
			delete layers[i];
		}
	}
	free(layers);
}


void THierarchicalPartition::computeHierarchicalMasses(double *mu, double **muLayers) {
	int l,x,y;
	// copy lowest layer
	for(x=0;x<layers[nLayers-1]->nCells;x++) {
		muLayers[nLayers-1][x]=mu[x];
	}
	// now go up through higher layers, each time summing mass of children
	l=nLayers-2;
	while(l>=0) {
		// go through all cells of current layer
		for(x=0;x<layers[l]->nCells;x++) {
			// reset mass of current cell
			muLayers[l][x]=0;
			// now go through children of this cell
			for(y=0;y<layers[l]->nChildren[x];y++) {
				muLayers[l][x]+=muLayers[l+1][
					layers[l]->children[x][y]
					];
			}
		}
		l--;
	}
}

double** THierarchicalPartition::signal_allocate_double(int lTop, int lBottom) {
	double **result;
	result=(double**) malloc(sizeof(double*)*(lBottom-lTop+1));
	for(int i=0;i<lBottom-lTop+1;i++) {
		result[i]=(double*) calloc(layers[lTop+i]->nCells,sizeof(double));
	}
	
	return result;
}

void THierarchicalPartition::signal_free_double(double **signal, int lTop, int lBottom) {
	for(int i=0;i<=lBottom-lTop;i++) {
		free(signal[i]);
	}
	free(signal);

}

void THierarchicalPartition::signal_propagate_double(double **signal, const int lTop, const int lBottom, const int mode) {
	double newValue,value;
	// iterate through all involved layers (except the finest one), from bottom up
	for(int i=lBottom-1;i>=lTop;i--) {
		// go through all cells on current layer
		for(int j=0;j<layers[i]->nCells;j++) {
			// initialize signal with value of first child
			value=signal[i-lTop+1][layers[i]->children[j][0] ];
			// iterate over rest of children
			for(int k=1;k<layers[i]->nChildren[j];k++) {
				newValue=signal[i-lTop+1][layers[i]->children[j][k] ];
				if( ((mode==MODE_MAX) && (newValue>value)) || ((mode==MODE_MIN) && (newValue<value))) {
					value=newValue;
				}
			}
			signal[i-lTop][j]=value;
		}
	}
}


void THierarchicalPartition::signal_propagate_int(int **signal, const int lTop, const int lBottom, const int mode) {
	int newValue,value;
	// iterate through all involved layers (except the finest one), from bottom up
	for(int i=lBottom-1;i>=lTop;i--) {
		// go through all cells on current layer
		for(int j=0;j<layers[i]->nCells;j++) {
			// initialize signal with value of first child
			value=signal[i-lTop+1][layers[i]->children[j][0] ];
			// iterate over rest of children
			for(int k=1;k<layers[i]->nChildren[j];k++) {
				newValue=signal[i-lTop+1][layers[i]->children[j][k] ];
				if( ((mode==MODE_MAX) && (newValue>value)) || ((mode==MODE_MIN) && (newValue<value))) {
					value=newValue;
				}
			}
			signal[i-lTop][j]=value;
		}
	}
}


void THierarchicalPartition::signal_refine_double(double *signal, double *signalFine, int lTop, int mode) {
	switch(mode) {
	case INTERPOLATION_MODE_GRIDLINEAR:
		signal_refine_double_gridlinear(signal, signalFine, lTop);
		break;
	default:
		signal_refine_double_constant(signal, signalFine, lTop);
		break;
	}
}


void THierarchicalPartition::signal_refine_double_constant(double *signal, double *signalFine, int lTop) {
	int x,xFineIndex,xFine;
	// iterate over top layer
	for(x=0;x<layers[lTop]->nCells;x++) {
		// iterate over children
		for(xFineIndex=0;xFineIndex<layers[lTop]->nChildren[x];xFineIndex++) {
			xFine=layers[lTop]->children[x][xFineIndex];
			// set signal to parent value
			signalFine[xFine]=signal[x];
		}
	}
}

void THierarchicalPartition::signal_refine_double_gridlinear(double *signal, double *signalFine, int lTop) {

	// compute linear interpolation of old signal
	// this only works if the partition is a regular 2^dim-tree in dim=2, with child-mode CM_Grid.
	// such that the indices of the points indicate their position on the grid

	// number of grid points along each axis
	int nCoarse=round(pow(2,lTop));
	int nFine=nCoarse*2;
	
	// get shape and strides of fine and coarse grid
	std::vector<int> dims(dim,nFine);
	std::vector<int> dimsCoarse(dim,nCoarse);
	std::vector<int> strides(dim);
	std::vector<int> stridesCoarse(dim);
	GridToolsGetStrides(dim, dims.data(), strides.data());
	GridToolsGetStrides(dim, dimsCoarse.data(), stridesCoarse.data());
	
	
	int x; // fine index
	std::vector<int> y(dim); // grid indices of fine grid point
	// the fine grid point lies in a cubic cell spanned by some coarse grid points. following variables store their indices
	// at the grid boundaries one needs to be a bit careful
	std::vector<int> yCoarsePre(dim); // grid indices of coarse grid point on "top left"
	std::vector<int> yCoarseNext(dim); // grid indices of fine grid point on "bottom right"
	std::vector<double> w(dim); // relative weights for interpolation of carse pre and next grid point along each axis


	// iterate over fine layer
	for(x=0;x<layers[lTop+1]->nCells;x++) {
		//printf("x: %d\n",x);
		// compute grid coordinates
		GridToolsGetPosFromId(dim, x, y.data(), strides.data());
		//printf("\tpos: ");
		//for(int d=0;d<dim; d++) {
		//	printf("%d ",y[d]);
		//}
		//printf("\n");

		//printf("\tcoords and weights: ");
		// compute coarse coordinates and weights
		for(int d=0;d<dim;d++) {
			if(y[d]==0) {
				yCoarsePre[d]=0;
				yCoarseNext[d]=0;
				w[d]=0.;
			} else if (y[d]==dims[d]-1) {
				yCoarsePre[d]=dimsCoarse[d]-1;
				yCoarseNext[d]=0;
				w[d]=1.;
			} else {
				yCoarsePre[d]=(y[d]-1)/2;
				yCoarseNext[d]=yCoarsePre[d]+1;
				w[d]=1.25-0.5*(y[d]-2*yCoarsePre[d]);
			}
			//printf("  (%d  %d  %f)  ",yCoarsePre[d],yCoarseNext[d],w[d]);
		}
		//printf("\n");
		
		// reset result to zero
		signalFine[x]=0.;

		
		// now iterate over all surrounding grid points
		int nNeighbours=round(pow(2,dim)); // number of neighbours (2 per dim)
		// do for loop over all neighbours enumerated, then compute pos by mod and div wrt powers of two
		for(int nId=0;nId<nNeighbours;nId++) {
			//printf("\t\t%d\t",nId);
			int xCoarse=0; // id of current neighbour
			double wCoarse=1.; // weight of current neighbour
			for(int d=0;d<dim;d++) {
				int coordPreNext=((int) round(nId/pow(2,d)))%2; // if along this dim need pre or next
				// depending on this, compute contribution of d-th axis to id and weight of this neighbour
				if(coordPreNext==0) {
					xCoarse+=stridesCoarse[d]*yCoarsePre[d];
					wCoarse*=w[d];
				} else {
					xCoarse+=stridesCoarse[d]*yCoarseNext[d];
					wCoarse*=(1-w[d]);
				}
				//printf("  (%d %f)  ",xCoarse,wCoarse);
			}
			signalFine[x]+=wCoarse*signal[xCoarse];
			//printf("\n");
		}


	}
	
	
}


template<typename T>
THierarchicalProductSignal<T>::THierarchicalProductSignal(THierarchicalPartition *_partitionX, THierarchicalPartition *_partitionY) {
	partitionX=_partitionX;
	partitionY=_partitionY;
}


template<typename T>
void THierarchicalProductSignal<T>::signal_propagate(T **signal, int lTop, int lBottom, int mode) {
	T newValue,value;
	TPartitionLayer *layerX, *layerY;
	int yres;
	int x,y,xFineIndex,yFineIndex,xFine,yFine;
	// iterate through all involved layers (except the finest one), from bottom up
	for(int i=lBottom-1;i>=lTop;i--) {
		layerX=partitionX->layers[i];
		layerY=partitionY->layers[i];
		yres=partitionY->layers[i+1]->nCells;
		// go through all cells on current layer in X
		for(x=0;x<(int) layerX->nCells;x++) {
			// go through all cells on current layer in Y
			for(y=0;y<(int) layerY->nCells;y++) {
				value=0;
				// iterate over children
				for(xFineIndex=0;xFineIndex<(int) layerX->nChildren[x];xFineIndex++) {
					xFine=layerX->children[x][xFineIndex];
					for(yFineIndex=0;yFineIndex<(int) layerY->nChildren[y];yFineIndex++) {
						yFine=layerY->children[y][yFineIndex];
						newValue=signal[i-lTop+1][xFine*yres + yFine ];
						if(((xFineIndex==0) && (yFineIndex==0))
								|| ((mode==MODE_MAX) && (newValue>value))
								|| ((mode==MODE_MIN) && (newValue<value))
								) {
							value=newValue;
						}
					}
				}
				signal[i-lTop][x*layerY->nCells+y]=value;
			}
		}
	}
}

template<typename T>
TVarListHandler* THierarchicalProductSignal<T>::check_dualConstraints(T **_c, T **_alpha, T **_beta, int lTop, int lBottom, T _slack) {
	TVarListHandler* result;
	int xres,x,yres,y;
	
	// store params to temporariy global variables
	c=_c;
	alpha=_alpha;
	beta=_beta;
	slack=_slack;
	
	// create varList
	xres=partitionX->layers[lBottom]->nCells;
	result=new TVarListHandler();
	result->setupEmpty(xres);
	
	// set global varList
	varList=result;
	
	// initialize iterations
	xres=partitionX->layers[lTop]->nCells;
	yres=partitionY->layers[lTop]->nCells;
	for(x=0;x<xres;x++) {
		for(y=0;y<yres;y++) {
			check_dualConstraints_iterateTile(lTop, x, y, lBottom);
		}
	}
	
	// reset global variables
	c=NULL;
	alpha=NULL;
	beta=NULL;
	varList=NULL;
	
	// return
	return result;
}

template<typename T>
void THierarchicalProductSignal<T>::check_dualConstraints_iterateTile(int l, int x, int y, int lBottom) {
	// analyze cell (x,y) on layer l. That is go through all its children on layer l+1, check constraints there.
	// if violated either refine (if l+1<lBottom) or add variable
	int yres,xFine,yFine,xFineIndex,yFineIndex;
	
	//xres=partitionX->layers[l+1]->nCells;
	yres=partitionY->layers[l+1]->nCells;
	
	for(xFineIndex=0;xFineIndex<(int) partitionX->layers[l]->nChildren[x];xFineIndex++) {
		xFine=partitionX->layers[l]->children[x][xFineIndex];
		
		for(yFineIndex=0;yFineIndex<(int) partitionY->layers[l]->nChildren[y];yFineIndex++) {
			yFine=partitionY->layers[l]->children[y][yFineIndex];
			if(c[l+1][xFine*yres+yFine]-alpha[l+1][xFine]-beta[l+1][yFine]<=slack) {
				if(l+1==lBottom) {
					varList->addToLine(xFine,yFine,false);
				} else{
					check_dualConstraints_iterateTile(l+1, xFine, yFine, lBottom);
				}
			}
		}
	}
	
}



template<typename T>
TVarListHandler* THierarchicalProductSignal<T>::check_dualConstraints_adaptive(T **_c, T **_alpha, T **_beta, int lTop, int lBottom,
		T **_slackOffsetX, T **_slackOffsetY) {
	TVarListHandler* result;
	int xres,x,yres,y;
	
	// store params to temporary global variables
	c=_c;
	alpha=_alpha;
	beta=_beta;
	slackOffsetX=_slackOffsetX;
	slackOffsetY=_slackOffsetY;
	
	// create varList
	xres=partitionX->layers[lBottom]->nCells;
	result=new TVarListHandler();
	result->setupEmpty(xres);
	
	// set global varList
	varList=result;
	
	// initialize iterations
	xres=partitionX->layers[lTop]->nCells;
	yres=partitionY->layers[lTop]->nCells;
	for(x=0;x<xres;x++) {
		for(y=0;y<yres;y++) {
			check_dualConstraints_adaptive_iterateTile(lTop, x, y, lBottom);
		}
	}
	
	// reset global variables
	c=NULL;
	alpha=NULL;
	beta=NULL;
	varList=NULL;
	slackOffsetX=NULL;
	slackOffsetY=NULL;
	
	// return
	return result;
}

template<typename T>
void THierarchicalProductSignal<T>::check_dualConstraints_adaptive_iterateTile(int l, int x, int y, int lBottom) {
	// analyze cell (x,y) on layer l. That is go through all its children on layer l+1, check constraints there.
	// if violated either refine (if l+1<lBottom) or add variable
	int yres,xFine,yFine,xFineIndex,yFineIndex;
	double slackValue;
	
	//xres=partitionX->layers[l+1]->nCells;
	yres=partitionY->layers[l+1]->nCells;
	
	for(xFineIndex=0;xFineIndex<(int) partitionX->layers[l]->nChildren[x];xFineIndex++) {
		xFine=partitionX->layers[l]->children[x][xFineIndex];
		
		for(yFineIndex=0;yFineIndex<(int) partitionY->layers[l]->nChildren[y];yFineIndex++) {
			yFine=partitionY->layers[l]->children[y][yFineIndex];
			slackValue=c[l+1][xFine*yres+yFine]-alpha[l+1][xFine]-beta[l+1][yFine];
			if((slackValue<=slackOffsetX[l+1][xFine]) || (slackValue<=slackOffsetY[l+1][yFine])) {
				//		<< slackOffsetX[l+1][xFine] << "\t" << slackOffsetY[l+1][yFine] << "\n" << endl;
				if(l+1==lBottom) {
					varList->addToLine(xFine,yFine,false);
				} else{
					check_dualConstraints_adaptive_iterateTile(l+1, xFine, yFine, lBottom);
				}
			}
		}
	}
	
}


template class THierarchicalProductSignal<double>;


TVarListHandler* refineVarList(THierarchicalPartition *partitionX, THierarchicalPartition *partitionY,
		TVarListHandler *varListCoarse, int layerIdCoarse, bool doSort) {

	TPartitionLayer *lXC, *lYC;
	lXC=partitionX->layers[layerIdCoarse];
	lYC=partitionY->layers[layerIdCoarse];
	
	TVarListHandler *result=new TVarListHandler();
	result->setupEmpty(partitionX->layers[layerIdCoarse+1]->nCells);
	
	for(int xC=0;xC<lXC->nCells;xC++) {
		for(int yIdC=0;yIdC<varListCoarse->lenList[xC];yIdC++) {
			int yC=varListCoarse->varList[xC][yIdC];
			for(int xId=0;xId<lXC->nChildren[xC];xId++) {
				int x=lXC->children[xC][xId];
				for(int yId=0;yId<lYC->nChildren[yC];yId++) {
					int y=lYC->children[yC][yId];
					result->addToLine(x,y,false);
				}
			}
		}
	}
	
	if(doSort) {
		result->sort();
	}
	
	return result;

}

