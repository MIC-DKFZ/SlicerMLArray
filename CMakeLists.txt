cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerMLArray)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/MIC-DKFZ/SlicerMLArray")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Karol Gotkowski (DKFZ)")
set(EXTENSION_DESCRIPTION "Open and save MLArray (.mla) files: images as volumes, label maps as segmentations with names, colours and codes, and bounding boxes as ROIs, placed exactly by the file's geometry.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/MIC-DKFZ/SlicerMLArray/main/SlicerMLArray.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/MIC-DKFZ/SlicerMLArray/main/docs/screenshot.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerMLArray)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
