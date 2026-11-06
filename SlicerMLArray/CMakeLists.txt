#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerMLArray)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/boxes.py
  ${MODULE_NAME}Lib/dependencies.py
  ${MODULE_NAME}Lib/errors.py
  ${MODULE_NAME}Lib/export.py
  ${MODULE_NAME}Lib/geometry.py
  ${MODULE_NAME}Lib/layout.py
  ${MODULE_NAME}Lib/slicer_io.py
  ${MODULE_NAME}Lib/values.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
