#include"HK.h"

template<class TGeometry>
TParticleContainer ModelHK_Interpolate(const TSparsePosContainer& couplingData,
		const double * const muXEff, const double * const muYEff,
		const double * const muX, const double * const muY,
		const double * const posX, const double * const posY,
		const int dim, const double t, const double HKscale,
		const TGeometry& geometry) {
	// compute displacement interpolation for HK distance
	
	// couplingData contains coupling data as TSparsePosContainer.
	//  couplingData.mass: particle masses,
	//  couplingData.posStart and posEnd: indices of initial and final position
	
	// muXEff, muYEff: marginals of coupling
	// muX, muY: actual prescribed marginals
	
	// posX, posY: spatial position of marginal particles
	// dim: spatial dimension
	// t: time in [0,1] for which to compute interpolation snapshot
	
	// returns TParticleContainer
	//  mass: masses of particles
	//  pos: spatial positions of particles
	
	TParticleContainer result;
	result.dim=dim;
		
	// special case: at intial time
	if(t==0) {
		// set number of particles: initial marginal
		result.nParticles=couplingData.xres;

		// allocate mass and position array
		result.mass.resize(result.nParticles);
		result.pos.resize(result.nParticles*dim);

		// copy mass from muX
		std::copy(muX,muX+couplingData.xres,result.mass.data());
		// copy positions from posX
		std::copy(posX,posX+couplingData.xres*dim,result.pos.data());
		
		return result;
	}
	
	// special case: at final time
	if(t==1) {
		// set number of particles: initial marginal
		result.nParticles=couplingData.yres;

		// allocate mass and position array
		result.mass.resize(result.nParticles);
		result.pos.resize(result.nParticles*dim);

		// copy mass from muY
		std::copy(muY,muY+couplingData.yres,result.mass.data());
		// copy positions from posY
		std::copy(posY,posY+couplingData.yres*dim,result.pos.data());
		
		return result;
	}
	
	//////////////////////////////////////////////////////////////////////////////////
	// intermediate times: use standard formula
	
	// first determine number of "teleporting" particles. one for each marginal entry where mu(X/Y)Eff is zero
	int nParticlesTeleportX=0;
	for(int x=0;x<couplingData.xres;x++) {
		if(muXEff[x]==0) {
			nParticlesTeleportX+=1;
		}
	}
	int nParticlesTeleportY=0;
	for(int y=0;y<couplingData.yres;y++) {
		if(muYEff[y]==0) {
			nParticlesTeleportY+=1;
		}
	}
	
	
	// set number of particles: travelling particles (one for each entry in couplingData) and teleporting particles
	result.nParticles=couplingData.nParticles+nParticlesTeleportX+nParticlesTeleportY;

	// allocate mass and position array
	result.mass.resize(result.nParticles);
	result.pos.resize(result.nParticles*dim);

	
	// iterate over moving particles and set result to HK interpolation
	for(int i=0;i<couplingData.nParticles;i++) {
		// compute distance between start and end point
		double dist=geometry.dist(posX+(couplingData.posStart[i]*dim), posY+(couplingData.posEnd[i]*dim),dim);
		
		// initial and final mass of particle (coupling mass * density of respective marginal)
		double m0=couplingData.mass[i]*muX[couplingData.posStart[i]]/muXEff[couplingData.posStart[i]];
		double m1=couplingData.mass[i]*muY[couplingData.posEnd[i]]/muYEff[couplingData.posEnd[i]];
		
		// intermediate mass of particle at time t
		double mT=std::pow(1-t,2)*m0 + std::pow(t,2)*m1+2*t*(1-t)*std::sqrt(m0*m1)*std::cos(dist/HKscale);
		
		// set interpolated mass in result
		result.mass[i]=mT;
		
		
		// interpolate location
		if(dist>0) {
			// if actually moving
			
			// "angle" at intermediate position
			double phi=std::acos(
					((1-t)*std::sqrt(m0)+t*std::sqrt(m1)*std::cos(dist/HKscale))/
					std::sqrt(mT)
					);
			// transform into relative position along geodesic (in un-scaled setting phi equals the distance that has been travelled)
			phi=phi/dist*HKscale;
			// invoke call to geodesic computer to transform relative position into position
			geometry.geodesic(
				posX+(couplingData.posStart[i]*dim),
				posY+(couplingData.posEnd[i]*dim),
				result.pos.data()+(i*dim),
				dim,phi);
		} else {
			// if initial and final location coincide (dist==0)
			// just copy
			std::copy(
					posX+(couplingData.posStart[i]*dim),
					posX+((couplingData.posStart[i]+1)*dim),
					result.pos.data()+(i*dim)
					);
		}

	}
	
	// teleporting particles X
	int i=couplingData.nParticles; // keeps track of position in interpolating particle list
	for(int x=0;x<couplingData.xres;x++) {
		if(muXEff[x]==0) {
			// if no mass at this x, this x corresponds to a teleporting particle
			result.mass[i]=muX[x]*pow(1-t,2); // mass is reduced quadratically
			std::copy(posX+(x*dim),posX+((x+1)*dim),result.pos.data()+(i*dim)); // copy x location
			i++; // increase particle index
		}
	}


	// teleporting particles Y
	for(int y=0;y<couplingData.yres;y++) {
		if(muYEff[y]==0) {
			// if no mass at this y, this y corresponds to a teleporting particle
			result.mass[i]=muY[y]*pow(t,2); // mass is grown quadratically
			std::copy(posY+(y*dim),posY+((y+1)*dim),result.pos.data()+(i*dim)); // copy x location
			i++; // increase particle index
		}
	}
		
	return result;
}

template TParticleContainer ModelHK_Interpolate<TGeometry_Euclidean>(const TSparsePosContainer& couplingData,
		const double * const muXEff, const double * const muYEff,
		const double * const muX, const double * const muY,
		const double * const posX, const double * const posY,
		const int dim, const double t, const double kappa,
		const TGeometry_Euclidean& geometry);


