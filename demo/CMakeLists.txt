# Install data and configurations

set(CTR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/CTR/CTR.conf
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/CTR/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/CTR/train.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/CTR/test.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/CTR/run.sh)

set(MUSHROOM_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Mushroom/mushroom.conf
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Mushroom/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Mushroom/test.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Mushroom/train.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Mushroom/run.sh)

set(TITANIC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Titanic/titanic.conf
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Titanic/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Titanic/test.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Titanic/train.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_classification/Titanic/run.sh)

install(FILES ${CTR_FILES} DESTINATION bin/demo/CTR)
install(FILES ${MUSHROOM_FILES} DESTINATION bin/demo/Mushroom)
install(FILES ${TITANIC_FILES} DESTINATION bin/demo/Titanic)
