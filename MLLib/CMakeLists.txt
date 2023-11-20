cmake_minimum_required(VERSION 3.28)

project(MLLib VERSION 1.0 LANGUAGES CXX)

set(sources
	source/MLLib.cpp

	include/MLLib/TypeDefinitions.h
	include/MLLib/Regressor.h
	include/MLLib/RegressionTypes.h
	include/MLLib/KernelTypes.h
	include/MLLib/ModifierTypes.h
	include/MLLib/PrincipalComponentAnalysis.h

	include/MLLib/impl/Regressor.hpp
	include/MLLib/impl/RegressionTypes.hpp
	include/MLLib/impl/KernelTypes.hpp
	include/MLLib/impl/ModifierTypes.hpp
	include/MLLib/impl/PrincipalComponentAnalysis.hpp
)

add_library(${PROJECT_NAME} ${sources})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} 
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/include/MLLib
)

add_subdirectory(ThirdParty/dlib dlib)

target_link_libraries(${PROJECT_NAME}
		PUBLIC
			dlib
)


