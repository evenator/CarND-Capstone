cmake_minimum_required(VERSION 2.8.3)
project(tl_detector)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
        scripts/light_publisher
        scripts/tl_detector
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

