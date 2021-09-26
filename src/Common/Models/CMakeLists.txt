add_library(Models OBJECT
	OT.cpp HK.cpp
	Geometry_Sphere.cpp Geometry_Reflector.cpp
	)

install (FILES
	Geometry_Reflector.h
	Geometry_Sphere.h
	OT.h
	TGeometry.h
	HK.h
	DESTINATION ${CMAKE_BINARY_DIR}/../include/Common/Models)
