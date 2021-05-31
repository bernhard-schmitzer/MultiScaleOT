cmake_minimum_required(VERSION 3.0)

# project name
project (MultiScaleOT)

# C++11 is required to build this project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

# add src root as include directory
include_directories(.)

# Position Independent Code
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Misc compiler flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -pedantic-errors")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pedantic-errors")


# Verbose mode
option(SET_VERBOSE "Verbose output" ON)
if( SET_VERBOSE )
	set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
endif ( SET_VERBOSE )



# configure CPLEX
option(USE_CPLEX "Use the CPLEX Network Flow Solver as backend" OFF)
set(CPLEX_LIBRARY "" CACHE FILEPATH "Location of CPLEX binaries for linking")
set(CPLEX_INCLUDE_DIRECTORY "" CACHE PATH "Location for CPLEX headers for compilation")

if( USE_CPLEX )
        set(CPLEX_COMPILE_OPTIONS -DNDEBUG -DIL_STD -DILOSTRICTPOD)
        # define USE_CPLEX compiler flag
        set(CPLEX_COMPILE_OPTIONS ${CPLEX_COMPILE_OPTIONS} -DUSE_CPLEX)
        set(CPLEX_LIBRARIES ${CPLEX_LIBRARY} pthread)

        add_subdirectory (LP_CPLEX)

endif ( USE_CPLEX )



# configure Lemon
option(USE_LEMON "Use the Lemon Network Flow Solver as backend" OFF)
set(LEMON_LIBRARY "" CACHE FILEPATH "Location of LEMON binaries for linking")
set(LEMON_INCLUDE_DIRECTORY "" CACHE PATH "Location of LEMON headers for compilation")

if( USE_LEMON )
        set(LEMON_COMPILE_OPTIONS "")
        set(LEMON_COMPILE_OPTIONS ${LEMON_COMPILE_OPTIONS} -DUSE_LEMON)


        add_subdirectory (LP_Lemon)

endif ( USE_LEMON )


# configure eigen3
find_package (Eigen3 3.3 QUIET)
if( EIGEN3_FOUND )
	message(STATUS "Eigen3 library found at ${EIGEN3_INCLUDE_DIR}")
else(EIGEN3_FOUND)
	message(STATUS "Eigen3 library not found")
endif(EIGEN3_FOUND)

# configure Sinkhorn
option(USE_SINKHORN "Compile Sinkhorn solver" ON)
if( USE_SINKHORN )
	if(NOT EIGEN3_FOUND)
		message(FATAL_ERROR "Need Eigen3 library for Sinkhorn module.")
	endif(NOT EIGEN3_FOUND)
	set(SINKHORN_COMPILE_OPTIONS "")
	set(SINKHORN_INCLUDE_DIRECTORY ${EIGEN3_INCLUDE_DIR})
	add_subdirectory (Sinkhorn)
endif ( USE_SINKHORN )



# configure pybind11 interface
option(USE_PYBIND "Compile pybind11 interface" ON)

#set(PYTHON_PATH "/usr/bin" CACHE PATH "Location of python3 executable for python interface compilation")
#get_filename_component(PYTHON_PATH "${PYTHON_EXECUTABLE}" DIRECTORY)

if( USE_PYBIND )

	add_subdirectory(pybind11)
	#exec_program("${PYTHON_PATH}/python3-config"
	#		ARGS "--extension-suffix"
	#		OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
	#		RETURN_VALUE PYTHON_EXTENSION_SUFFIX_NOT_FOUND
	#		)

	#if(NOT PYTHON_EXTENSION_SUFFIX_NOT_FOUND EQUAL 0)
	#	message(FATAL_ERROR "Did not find python3-config for extension suffix.")
	#endif(NOT PYTHON_EXTENSION_SUFFIX_NOT_FOUND EQUAL 0)

	#exec_program("${PYTHON_PATH}/python3-config"
	#		ARGS "--includes"
	#		OUTPUT_VARIABLE PYTHON_INCLUDES
	#		RETURN_VALUE PYTHON_INCLUDES_NOT_FOUND
	#		)

	#if(NOT PYTHON_INCLUDES_NOT_FOUND EQUAL 0)
	#	message(FATAL_ERROR "Did not find python3-config for includes.")
	#endif(NOT PYTHON_INCLUDES_NOT_FOUND EQUAL 0)


	#separate_arguments(PYTHON_INCLUDES UNIX_COMMAND "${PYTHON_INCLUDES}")

	#set(PYBIND_COMPILE_OPTIONS ${PYTHON_INCLUDES})
	set(PYBIND_COMPILE_OPTIONS "")
	if( USE_SINKHORN )
		set(PYBIND_COMPILE_OPTIONS ${PYBIND_COMPILE_OPTIONS} -DUSE_SINKHORN)
	endif( USE_SINKHORN )
	if( USE_CPLEX )
		set(PYBIND_COMPILE_OPTIONS ${PYBIND_COMPILE_OPTIONS} -DUSE_CPLEX)
	endif( USE_CPLEX )


	add_subdirectory (pybind11_interface)
endif ( USE_PYBIND )


# find all project files
add_subdirectory (Common)
add_subdirectory (ShortCutSolver)
add_subdirectory (Examples)


install (FILES Common.h LP_CPLEX.h LP_Lemon.h ShortCutSolver.h Sinkhorn.h DESTINATION ${CMAKE_BINARY_DIR}/../include)
