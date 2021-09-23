#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Common/Verbose.h>

#ifdef USE_CPLEX
#include<LP_CPLEX.h>
#endif

#include<pybind11/include/pybind11/pybind11.h>
#include<pybind11/include/pybind11/numpy.h>

#include"MultiScaleOT_pybind11_interface_common.h"

namespace py = pybind11;


using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_CPLEX
void init_cplex(py::module_ &m) {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    // experimental / preliminary CPLEX LP solver interface
    m.def("CPLEX_OT_solveDense",
            [](py::array_t<double> &c, py::array_t<double> &muX, py::array_t<double> &muY) {
                int msg;
                
                py::buffer_info bufferC = c.request();
                py::buffer_info bufferX = muX.request();
                py::buffer_info bufferY = muY.request();

                int xres=bufferC.shape[0];
                int yres=bufferC.shape[1];
 
                std::vector<double> pi(xres*yres);
                std::vector<double> alpha(xres);
                std::vector<double> beta(yres);

                TCouplingHandlerDense couplingHandler(xres,yres,(double *) bufferC.ptr,pi.data());

                TCPLEXNetSolver<TCouplingHandlerDense> solver(&couplingHandler,(double*) bufferX.ptr,(double*) bufferY.ptr,alpha.data(),beta.data());

                msg=solver.setup();
                if(msg!=0) {
                    return py::make_tuple(msg);
                }
                msg=solver.solve();
                if(msg!=0) {
                    return py::make_tuple(msg);
                }

                std::vector<int> dimensions={xres,yres};
                return py::make_tuple(0,solver.getObjective(),
                        getArrayFromRaw<double>(pi.data(), dimensions),
                        getArrayFromRaw<double>(alpha.data(), xres),
                        getArrayFromRaw<double>(beta.data(), yres)
                        );

	

            },
            "Solve a dense discrete optimal transport problem with the CPLEX network flow solver.",
            py::arg("c"),py::arg("muX"),py::arg("muY")
            );
    m.def("CPLEX_network_flow",
            [](py::array_t<double> &node_supply,
                    py::array_t<int> &arc_from, py::array_t<int> &arc_to,
                    py::array_t<double> &arc_low, py::array_t<double> &arc_up,
                    py::array_t<double> &arc_cost) {
                int msg;
                
                
                py::buffer_info buffer_node_supply = node_supply.request();
                py::buffer_info buffer_arc_from = arc_from.request();
                py::buffer_info buffer_arc_to = arc_to.request();
                py::buffer_info buffer_arc_low = arc_low.request();
                py::buffer_info buffer_arc_up = arc_up.request();
                py::buffer_info buffer_arc_cost = arc_cost.request();

                int nNodes=buffer_node_supply.shape[0];
                int nArcs=buffer_arc_from.shape[0];
 
                double *lowptr=NULL;
                double *upptr=NULL;
                if (buffer_arc_low.shape[0]==nArcs) lowptr=(double*) buffer_arc_low.ptr;
                if (buffer_arc_up.shape[0]==nArcs) upptr=(double*) buffer_arc_up.ptr;
 
                std::vector<double> flow(nArcs);
                std::vector<double> potential(nNodes);


                // now use CPLEX network flow solver interface
                CPXENVptr CPXenv = CPXXopenCPLEX(&msg);
                CPXNETptr CPXnet = CPXXNETcreateprob(CPXenv, &msg, NULL);

                // add nodes
                CPXXNETaddnodes(CPXenv, CPXnet, nNodes, (double*) buffer_node_supply.ptr, NULL);
                // add arcs
                CPXXNETaddarcs(CPXenv, CPXnet, nArcs, (int*) buffer_arc_from.ptr, (int*) buffer_arc_to.ptr, lowptr, upptr, (double*) buffer_arc_cost.ptr, NULL);

                msg=CPXXNETprimopt(CPXenv, CPXnet);
                if(msg!=0) {
    	            return py::make_tuple(1,msg);
	            }

                double objective;
                CPXXNETsolution(CPXenv, CPXnet, &msg, &objective, flow.data(), potential.data(), NULL, NULL);
                if(msg!=CPX_STAT_OPTIMAL) {
    	            return py::make_tuple(1,msg);
	            }
                
                return py::make_tuple(0,objective,
                        getArrayFromRaw<double>(flow.data(), nArcs),
                        getArrayFromRaw<double>(potential.data(), nNodes)
                        );

	

            },
            "Solve a network flow problem with the CPLEX network flow solver.",
            py::arg("node_supply"),py::arg("arc_from"),py::arg("arc_to"),py::arg("arc_low"),py::arg("arc_up"),py::arg("arc_cost")
            );
}
#endif

