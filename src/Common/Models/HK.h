#ifndef Model_HK_H_
#define Model_HK_H_

#include<algorithm>

#include<Common/Tools.h>

#include<Common/Models/OT.h>
#include<Common/Models/TGeometry.h>

template<class TGeometry>
TParticleContainer ModelHK_Interpolate(const TSparsePosContainer& couplingData,
		const double * const muXEff, const double * const muYEff,
		const double * const muX, const double * const muY,
		const double * const posX, const double * const posY,
		const int dim, const double t, const double HKscale,
		const TGeometry& geometry);


#endif
