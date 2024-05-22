# meta
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# project
project(CCF_QDALS)
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

# NOTE: QPanda install path, you may need to modify this!
set(QPANDA_INSTALL_DIR "/usr/local")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${QPANDA_INSTALL_DIR}/lib/cmake")
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# linker flags
set(LIB_LIST)
set(EXTRA_LDFLAGS)
if(MSVC)
    list(APPEND EXTRA_LDFLAGS
        -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
        -D_CRT_SECURE_NO_WARNINGS
        /bigobj)

    if(NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
        foreach(flag
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)

            if(${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MT" ${flag} "${${flag}}")
            endif()

            if(${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            
            if(${flag} MATCHES "/W3")
                string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
            endif()
        endforeach()
    endif()
elseif(MINGW)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fpermissive)

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -Og -ggdb -m64 -Wa, -mbig-obj")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
elseif(UNIX)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fPIC
        -fpermissive)

    if(NOT APPLE)
        list(APPEND LIB_LIST
            dl
            pthread)
    endif()

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
endif()
add_definitions(${EXTRA_LDFLAGS})

# QPanda
find_package(QPANDA REQUIRED)
if(QPANDA_FOUND)
    message("Find QPANDA_INCLUDE_DIR: " ${QPANDA_INCLUDE_DIR})
    message("Find QPANDA_LIBRARY: " ${QPANDA_LIBRARY})
    include_directories(${QPANDA_INCLUDE_DIR})
else(QPANDA_FOUND)
    message(FATAL_ERROR "Not find QPanda")
endif(QPANDA_FOUND)

# Curl
find_package(CURL REQUIRED)
if(CURL_FOUND)
    option(USE_CURL "find Curl" ON)
    message("Find CURL_INCLUDE_DIR: " ${CURL_INCLUDE_DIR})
	message("Find CURL_LIBRARY: " ${CURL_LIBRARY})
    include_directories(${CURL_INCLUDE_DIRS})
	if(MSVC)
	    set(CURL_LIBRARY libcurl-static)
    elseif(CMAKE_COMPILER_IS_GNUCXX)
	    set(CURL_LIBRARY curl)
    endif()
    list(APPEND LIB_LIST ${CURL_LIBRARY})
else(CURL_FOUND)
    option(USE_CURL "Not find Curl" OFF)
endif(CURL_FOUND)

# OpenMP
option(FIND_OPENMP "find OpenMP" ON)
if(FIND_OPENMP)
    find_package(OpenMP)
endif(FIND_OPENMP)
if(OPENMP_FOUND)
    option(USE_OPENMP "find OpenMP" ON)
    message("Find OPENMP_INCLUDE_DIR: " ${OPENMP_INCLUDE_DIR})
    message("Find OPENMP_LIBRARY: " ${OPENMP_LIBRARY})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    option(USE_OPENMP "Not find OpenMP" OFF)
endif(OPENMP_FOUND)

#include_directories(${PROJECT_SOURCE_DIR}/SU2/externals/cgns)
#add_definitions(-D HAVE_CGNS)
if(NOT CMAKE_BUILD_TYPE)
   message("No CMAKE_BUILD_TYPE, default to Debug")
   set(CMAKE_BUILD_TYPE "Debug")
endif()

# source path for all *.cpp/*.h
aux_source_directory(./src src)

# include paths for *.h
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/QSolver
    ${PROJECT_SOURCE_DIR}/QSolver/include
    ${PROJECT_SOURCE_DIR}/QSolverApplications
    ${QPANDA_INSTALL_DIR}/include
    ${QPANDA_INSTALL_DIR}/include/qpanda2
    ${QPANDA_INSTALL_DIR}/include/qpanda2/Extensions
    ${QPANDA_INSTALL_DIR}/include/qpanda2/ThirdParty
)

# executables
add_executable(
    ${PROJECT_NAME}
    src/utils.h
    src/utils.cpp
    src/block_encoding.h
    src/block_encoding.cpp
    src/qda_linear_solver.h
    src/qda_linear_solver.cpp
    src/main.cpp
)
target_link_libraries(${PROJECT_NAME} ${QPANDA_LIBRARIES} ${LIB_LIST})

add_executable(
    test_block_encoding
    src/utils.h
    src/utils.cpp
    src/block_encoding.h
    src/block_encoding.cpp
    src/block_encoding_test.cpp
)
target_link_libraries(test_block_encoding ${QPANDA_LIBRARIES} ${LIB_LIST})

#[[
add_executable(
    test_qda_linear_solver
    src/utils.h
    src/utils.cpp
    src/block_encoding.h
    src/block_encoding.cpp
    src/qda_linear_solver.h
    src/qda_linear_solver.cpp
    src/qda_linear_solver_test.cpp
)
target_link_libraries(test_qda_linear_solver ${QPANDA_LIBRARIES} ${LIB_LIST})

add_executable(
    test_matrix_decompose
    src/test_matrix_decompose.cpp
)
target_link_libraries(test_matrix_decompose ${QPANDA_LIBRARIES} ${LIB_LIST})
]]
