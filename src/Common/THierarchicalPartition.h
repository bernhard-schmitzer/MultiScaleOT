#ifndef THierarchicalPartition_H_
#define THierarchicalPartition_H_

#include<cmath>
#include<cstdlib>
#include<vector>
#include"Common/GridTools.h"
#include"Common/TVarListHandler.h"

class TPartitionLayer {
public:
	int nCells; // number of cells in that layer
	// only the parent-children-leaves structure is fundamental. the rest can be encoded as "signals" living on the partition
	// implement these signals separately
	int *parent;
	int **children, **leaves;
	int *nChildren, *nLeaves;
	TPartitionLayer();
	~TPartitionLayer();
	void initializeEmpty(int _nCells);
};

class THierarchicalPartition {
public:
	static const int MODE_MIN;
	static const int MODE_MAX;
	static const int INTERPOLATION_MODE_CONSTANT;
	static const int INTERPOLATION_MODE_GRIDLINEAR;
	
	TPartitionLayer **layers;
	int nLayers; // number of layers
	int dim; // dimension of ambient space
	int interpolation_mode; // how to refine duals from one layer to next?
	
	THierarchicalPartition(int _nLayers, int _dim, int _interpolation_mode);
	THierarchicalPartition(int _nLayers, int _dim) : THierarchicalPartition(_nLayers, _dim, INTERPOLATION_MODE_CONSTANT) {};
	~THierarchicalPartition();
	
	void computeHierarchicalMasses(double *mu, double **muLayers);
	double** signal_allocate_double(int lTop, int lBottom);
	void signal_free_double(double **signal, int lTop, int lBottom);
	void signal_propagate_double(double **signal, const int lTop, const int lBottom, const int mode);
	void signal_propagate_int(int **signal, const int lTop, const int lBottom, const int mode);
	
	void signal_refine_double(double *signal, double *signalFine, int lTop, int mode);
	void signal_refine_double(double *signal, double *signalFine, int lTop) {
		signal_refine_double(signal, signalFine, lTop, interpolation_mode);
	}
	void signal_refine_double_constant(double *signal, double *signalFine, int lTop);
	void signal_refine_double_gridlinear(double *signal, double *signalFine, int lTop);
};


template<typename T>
class THierarchicalProductSignal {
public:
	static const int MODE_MIN=0;
	static const int MODE_MAX=1;
	
	// temporariy "global" variables for check_dualConstraints
	T **c, **alpha, **beta;
	T slack;
	TVarListHandler* varList;
	// for advanced constraint checking (possibly need to clean this up later)
	T **slackOffsetX, **slackOffsetY;
	
	THierarchicalPartition *partitionX, *partitionY;
	THierarchicalProductSignal(THierarchicalPartition *_partitionX, THierarchicalPartition *_partitionY);
	void signal_propagate(T **signal, int lTop, int lBottom, int mode);
	TVarListHandler* check_dualConstraints(T **_c, T **_alpha, T **_beta, int lTop, int lBottom, T _slack);
	void check_dualConstraints_iterateTile(int l, int x, int y, int lBottom);
	//
	TVarListHandler* check_dualConstraints_adaptive(T **_c, T **_alpha, T **_beta, int lTop, int lBottom,
		T **_slackOffsetX, T **_slackOffsetY);
	void check_dualConstraints_adaptive_iterateTile(int l, int x, int y, int lBottom);
};


TVarListHandler* refineVarList(THierarchicalPartition *partitionX, THierarchicalPartition *partitionY,
		TVarListHandler *varListCoarse, int layerIdCoarse, bool doSort=false);

#endif
