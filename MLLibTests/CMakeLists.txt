cmake_minimum_required(VERSION 3.28)

project(MLLibTests VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(test_sources
	main.cpp
	OneShotRegressorTests.cpp
	CrossValidationRegressorTests.cpp
	FindMinGlobalRegressorTests.cpp
	RegressorTests.cpp
)

add_executable(RegressorTests ${test_sources})

add_subdirectory(ThirdParty/googletest)

set_target_properties(RegressorTests PROPERTIES FOLDER ${PROJECT_NAME})

target_link_libraries(RegressorTests 
	PRIVATE
		MLLib
		gtest
)
