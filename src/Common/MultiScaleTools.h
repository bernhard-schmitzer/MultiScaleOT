#ifndef MultiScaleTools_H_
#define MultiScaleTools_H_

#include<vector>

#include<Common/ErrorCodes.h>
#include<Common/Tools.h>
#include<Common/Verbose.h>

#include<Common/GridTools.h>
#include<Common/THierarchyBuilder.h>
#include<Common/THierarchicalPartition.h>
#include<Common/THierarchicalCostFunctionProvider.h>


// Algorithm for hierarchical nearest neighbour search

class THierarchicalNN {
public:
	struct TCandidate {
		int layer; // on which layer
		int z; // row/column index
		double dist; // lower bound on distance
		bool operator<(const THierarchicalNN::TCandidate& rhs) const { return dist < rhs.dist; };
	};
	
	typedef THierarchicalSearchList<THierarchicalNN::TCandidate> TCandidateList;


	static std::vector<int> find(double *posX, double **posYH, double **radiiY,
			THierarchicalPartition *HPY, int layerBottom, int nElements);
		// find nElement nearest neighbours of posX in the points of HPY
		// (with coordinates posYH and cell radii radiiY) at layer layerBottom
	
	static TVarListHandler* getNeighbours(double **posXH, double **radiiX,
			THierarchicalPartition *HPX, int layerBottom, int nElements);
		// find nElement nearest neighbours for each point in layerBottom of HPX
		// (with coordinates posXH, cell radii radiiX) by using the above find(...)
		// function on each point

	static TVarListHandler** getNeighboursH(double **posXH, double **radiiX,
			THierarchicalPartition *HPX, int nElements);
		// find nElement nearest neighbours for each point in layerBottom of HPX
		// (with coordinates posXH, cell radii radiiX) by using the above find(...)
		// function on each point

};


class THierarchicalDualMaximizer {
public:
	static constexpr int MODE_ALPHA=0;
	static constexpr int MODE_BETA=1;
	
	struct TCandidate {
		int layer; // on which layer
		int z; // row/column index
		double v; // uper bound on dual variable
		bool operator<(const THierarchicalDualMaximizer::TCandidate& rhs) const { return v < rhs.v; };
	};

	typedef THierarchicalSearchList<THierarchicalDualMaximizer::TCandidate> TCandidateList;

	static void getMaxDual(THierarchicalPartition *partitionX, THierarchicalPartition *partitionY,
			double **alpha, double **beta, int layerFine,
			THierarchicalCostFunctionProvider *costProvider,
			int mode);


};


// multi scale setup

class TMultiScaleSetup {
public:
	double *pos; // point clouds for marginal positions
	double *mu; // pointers to marginal masses
	int res; // integers for total cardinality of marginal points
	int dim; // dimensionality of marginal positions
	
	// hierarchy
	int depth; // parameter for controlling nr of layers in hierarchical partition
	int nLayers; // number of layers (usually depth+1)
	THierarchyBuilder *HB; // hierarchy builder classes
	THierarchicalPartition *HP; // hierarchical partition classes
	double **posH; // hierarchical positions
	double **muH; // hierarchical masses
	int *resH; // hierarchical marginal cardinality
	int HierarchyBuilderChildMode; // mode for handling children nodes in hierarchy builder
		// set to THierarchyBuilder::CM_Grid in constructor.
		
	// other hierarchical stuff (setup on demand)
	double **alphaH; // dual potentials
	double **radii; // radii for hiearchical partition cells
	// list of neighbour points for X
	TVarListHandler **neighboursH;
		
	//////////////////////////////////////////////////////////////////////////////
	
	TMultiScaleSetup(double *_pos, double *_mu, int _res, int _dim, int _depth,
			int _childMode,
			bool _setup, bool _setupDuals, bool _setupRadii
			);
	TMultiScaleSetup(double *_pos, double *_mu, int _res, int _dim, int _depth) : TMultiScaleSetup(_pos,_mu,_res,_dim,_depth,THierarchyBuilder::CM_Grid,true,true,true) {};
	TMultiScaleSetup(TDoubleMatrix *posMatrix, double *_mu, int _depth,
			bool _setup, bool _setupDuals, bool _setupRadii, int _childMode
			)  : TMultiScaleSetup(posMatrix->data, _mu, posMatrix->dimensions[0], posMatrix->dimensions[1], _depth, _childMode,
			_setup, _setupDuals, _setupRadii) {};
	TMultiScaleSetup(TDoubleMatrix *posMatrix, double *_mu, int _depth) : TMultiScaleSetup(posMatrix, _mu, _depth, THierarchyBuilder::CM_Grid,
			true, true, true) {};

	TMultiScaleSetup(const TMultiScaleSetup&) = delete;
	TMultiScaleSetup(TMultiScaleSetup&& b);
	
	virtual ~TMultiScaleSetup();

	int BasicMeasureChecks();
	virtual int SetupHierarchicalPartition();
	
	virtual int Setup();
	virtual int SetupDuals();
	virtual int SetupRadii();
	
	virtual int UpdatePositions(double *newPos);
	virtual int UpdateMeasure(double *newMu);
};



// Cartesian grid version

class TMultiScaleSetupGrid : public TMultiScaleSetup {
public:
	// multidimensional array of marginal measure
	TDoubleMatrix *muGrid;
	bool ownMuGrid; // whether muGrid needs to be freed upon destroying this class instance
	// grid dimensions of each hierarchy level. required for shield generators
	// this is a contiguous flattened 2d array with dimensions nLayers*dim
	int *dimH;
	TDoubleMatrix posExplicit; // stores explicit positions of all points, i.e. this class instance takes ownership of this
	
	TMultiScaleSetupGrid(TDoubleMatrix *_muGrid, int _depth, bool _setup, bool _setupDuals, bool _setupRadii);
	TMultiScaleSetupGrid(TDoubleMatrix *_muGrid, int _depth) : TMultiScaleSetupGrid(_muGrid, _depth, true, true, true) {};
	TMultiScaleSetupGrid(const TMultiScaleSetupGrid&) = delete;
	TMultiScaleSetupGrid(TMultiScaleSetupGrid&& b);

	int SetupHierarchicalPartition();
	int SetupGridNeighbours();
	// destructor
	virtual ~TMultiScaleSetupGrid();
	
};


class TMultiScaleSetupBarycenterContainer {
// class that provides arrays for relevant fields of several TMultiScaleSetup instances
// to use these in a SinkhornBarycenter solver
public:
	int nMarginals;
	THierarchicalPartition **HP,*HPZ; // hierarchical partition classes
	double ***muH,**muZH; // hierarchical masses
	double ***alphaH, ***betaH; // dual potentials
	int **resH, *resZH; // cardinalities of layers
	double *weights;
	THierarchicalCostFunctionProvider **costProvider;

	TMultiScaleSetupBarycenterContainer();
	TMultiScaleSetupBarycenterContainer(const int _nMarginals);
	~TMultiScaleSetupBarycenterContainer();
	void setupEmpty(const int _nMarginals);
	void cleanup();
	void setMarginal(const int n, TMultiScaleSetup &multiScaleSetup, const double weight);
	void setCenterMarginal(TMultiScaleSetup &multiScaleSetup);
	void setCostFunctionProvider(const int n, THierarchicalCostFunctionProvider &costFunctionProvider);
	void setupDuals();
};

#endif

