project( PCD_Viewer )
cmake_minimum_required( VERSION 2.8 )
add_executable(${PROJECT_NAME} Main.cpp PCL.cpp )

find_package( OpenGL )
find_package( OpenGL REQUIRED )
find_package( XercesC REQUIRED )
find_package( PCL 1.7 REQUIRED )

include( CheckCXXCompilerFlag )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

include_directories( ${PCL_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} /user/local/include )

link_directories( ${PCL_LIBRARY_DIRS} /usr/local/lib /usr/lib/x86_64-linux-gnu )
add_definitions( ${PCL_DEFINITIONS} )

IF( APPLE )
    # if using glfw on macOS, require below
    #  - cocoa.framework
    #  - OpenGL.framework
    #  - IOKit.framework
    #  - CoreVideo.framework
    find_library( COCOA_LIBRARY Cocoa )
    find_library( OpenGL_LIBRARY OpenGL )
    find_library( IOKIT_LIBRARY IOKit )
    find_library( COREVIDEO_LIBRARY CoreVideo )
    SET( EXTRA_LIBS ${COCOA_LIBRARY} ${OpenGL_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY} )
ENDIF( APPLE )

target_link_libraries( PCD_Viewer ${PCL_LIBRARIES} ${OPENGL_LIBRARY} ${XercesC_LIBRARIES} glfw ${EXTRA_LIBS} )
