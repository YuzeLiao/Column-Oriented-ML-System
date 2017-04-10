# Build library solver
add_library(train flags.cc train.cc f2m_dist_main.cc)

set(LIBS base train gflags reader thread updater loss data validator)

# Build execuation pragram
add_executable(f2m_main f2m_main.cc)
target_link_libraries(f2m_main ${LIBS})

# Install library and header files

install(TARGETS train DESTINATION lib/train)
install(TARGETS f2m_main DESTINATION bin)

FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
FILE(GLOB SCHE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/scheduler/*.py")

install(FILES ${HEADER_FILES} DESTINATION include/train)
install(FILES ${SCHE_FILE} DESTINATION bin/scheduler)

set(BIN_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/f2m.conf
    ${CMAKE_CURRENT_SOURCE_DIR}/run.py)

install(FILES ${BIN_FILES} DESTINATION bin)
