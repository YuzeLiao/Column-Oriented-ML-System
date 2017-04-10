# Build library thread.
add_library(thread condition_variable.cc)

# Build unittests.
set(LIBS thread base gtest pthread)

add_executable(condition_variable_test condition_variable_test.cc)
target_link_libraries(condition_variable_test gtest_main ${LIBS})

add_executable(mutex_test mutex_test.cc)
target_link_libraries(mutex_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS thread DESTINATION lib/thread)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/thread)
