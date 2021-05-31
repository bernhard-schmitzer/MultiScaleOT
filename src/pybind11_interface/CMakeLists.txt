if( USE_PYBIND )

	#add_library(pybind11_interface SHARED MultiScaleOT_pybind11_interface_common.cpp MultiScaleOT_pybind11_interface_base.cpp
	#	MultiScaleOT_pybind11_interface_sinkhorn.cpp MultiScaleOT_pybind11_interface_cplex.cpp)
	pybind11_add_module(pybind11_interface MODULE MultiScaleOT_pybind11_interface_common.cpp MultiScaleOT_pybind11_interface_base.cpp
		MultiScaleOT_pybind11_interface_sinkhorn.cpp MultiScaleOT_pybind11_interface_cplex.cpp)

	#target_include_directories(pybind11-interface PUBLIC ${PYBIND_INCLUDE_DIRECTORY})
	#target_include_directories(pybind11-interface PUBLIC ${EIGEN_LIBRARY})

	target_compile_options(pybind11_interface PUBLIC ${PYBIND_COMPILE_OPTIONS})
	
	target_link_libraries(pybind11_interface PRIVATE
		Common
		)
		
	if ( USE_SINKHORN )
		target_link_libraries(pybind11_interface PRIVATE
			Sinkhorn
			)
	endif ( USE_SINKHORN)

	if ( USE_CPLEX )
		target_include_directories(pybind11_interface PUBLIC ${CPLEX_INCLUDE_DIRECTORY})
		target_link_libraries(pybind11_interface PRIVATE
			LP_CPLEX
			${CPLEX_LIBRARIES})
	endif ( USE_CPLEX )
	
	#set_target_properties(pybind11_interface PROPERTIES PREFIX "")
	#set_target_properties(pybind11_interface PROPERTIES SUFFIX ${PYTHON_EXTENSION_SUFFIX})
	set_target_properties(pybind11_interface PROPERTIES OUTPUT_NAME MultiScaleOT)

	install (TARGETS pybind11_interface DESTINATION ${CMAKE_BINARY_DIR}/../python)


endif( USE_PYBIND )
