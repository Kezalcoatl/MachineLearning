cmake_minimum_required(VERSION 3.28)

project(MachineLearning VERSION 1.0 LANGUAGES CXX)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
	PRIVATE
		MachineLearning.cpp
)

set_property(TARGET MachineLearning PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
enable_testing()

add_subdirectory(MLLib)
add_subdirectory(MLLibTests)

target_link_libraries(${CMAKE_PROJECT_NAME}
		PUBLIC
			MLLib
)