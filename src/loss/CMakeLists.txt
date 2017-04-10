# Build library loss
add_library(loss loss.cc logit_loss.cc) #linear_loss.cc fm_loss.cc ffm_loss.cc svm_loss.cc)

# Build unittests.
set(LIBS data base loss gtest thread reader)

#add_executable(loss_test loss_test.cc)
#target_link_libraries(loss_test gtest_main ${LIBS})

#add_executable(logit_loss_test logit_loss_test.cc)
#target_link_libraries(logit_loss_test gtest_main ${LIBS})

#add_executable(linear_loss_test linear_loss_test.cc)
#target_link_libraries(linear_loss_test gtest_main ${LIBS})

#add_executable(svm_loss_test svm_loss_test.cc)
#target_link_libraries(svm_loss_test gtest_main ${LIBS})

#add_executable(fm_loss_test fm_loss_test.cc)
#target_link_libraries(fm_loss_test gtest_main ${LIBS})

#add_executable(ffm_loss_test ffm_loss_test.cc)
#target_link_libraries(ffm_loss_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS loss DESTINATION lib/loss)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/loss)
