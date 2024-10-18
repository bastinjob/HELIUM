configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(Helium_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        helium-opt
        )

add_lit_testsuite(check-Helium "Running the Helium regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${Helium_TEST_DEPENDS}
        )
set_target_properties(check-Helium PROPERTIES FOLDER "Tests")

add_lit_testsuites(Helium ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${Helium_TEST_DEPENDS})