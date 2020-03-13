if( USE_PYBIND )

	add_library(pybind11_interface SHARED MultiScaleOT_pybind11_interface.cpp)

	#target_include_directories(pybind11-interface PUBLIC ${PYBIND_INCLUDE_DIRECTORY})
	#target_include_directories(pybind11-interface PUBLIC ${EIGEN_LIBRARY})

	target_compile_options(pybind11_interface PUBLIC ${PYBIND_COMPILE_OPTIONS})
	
	target_link_libraries(pybind11_interface
		Common
		)
		
	if ( USE_SINKHORN )
		target_link_libraries(pybind11_interface
			Sinkhorn
			)
	endif ( USE_SINKHORN)
	
	set_target_properties(pybind11_interface PROPERTIES PREFIX "")
	set_target_properties(pybind11_interface PROPERTIES SUFFIX ${PYTHON_EXTENSION_SUFFIX})
	set_target_properties(pybind11_interface PROPERTIES OUTPUT_NAME MultiScaleOT)

	install (TARGETS pybind11_interface DESTINATION ${CMAKE_BINARY_DIR}/../python)


endif( USE_PYBIND )
