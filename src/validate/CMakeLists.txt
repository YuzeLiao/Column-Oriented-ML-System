# Build library validator
add_library(validator validator.cc)

# Install library and header files
install(TARGETS DESTINATION lib/validate)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/validator)
