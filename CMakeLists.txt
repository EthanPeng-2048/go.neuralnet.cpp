cmake_minimum_required(VERSION 3.30)

project(go_neuralnet
    VERSION 0.1.0
    DESCRIPTION "19x19 Go training and inference built on neuralnet.cpp"
    HOMEPAGE_URL "https://github.com/EthanPeng-2048/go.neuralnet.cpp"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# M0 consumes neuralnet.cpp as a source dependency. Keep the GPU switch at the
# project boundary so CPU-only CI and the 40HX benchmark use the same build file.
set(_go_gpu_default "${NN_ENABLE_GPU}")
if(NOT _go_gpu_default)
    set(_go_gpu_default "AUTO")
endif()
set(GO_ENABLE_GPU "${_go_gpu_default}" CACHE STRING "Vulkan GPU support: AUTO / ON / OFF")
set_property(CACHE GO_ENABLE_GPU PROPERTY STRINGS AUTO ON OFF)
set(NN_ENABLE_GPU "${GO_ENABLE_GPU}" CACHE STRING "Forwarded to neuralnet.cpp" FORCE)
unset(_go_gpu_default)

# layer_bench is an upstream application target. Defining it does not build it
# unless requested, so configuring all upstream apps keeps the M0 target graph
# ready without increasing the default go_neuralnet_m0 build.
option(GO_BUILD_UPSTREAM_TOOLS
       "Build upstream tools such as layer_bench for M0 benchmarks" ON)
if(GO_BUILD_UPSTREAM_TOOLS)
    set(NN_BUILD_APPS ON CACHE BOOL "Build neuralnet.cpp tools" FORCE)
endif()

add_subdirectory(3rd_party/neuralnet.cpp neuralnet_cpp)

add_executable(go_neuralnet_m0 src/main.cpp)
target_link_libraries(go_neuralnet_m0 PRIVATE neuralnet::nn)

# M1 uses KataGo's rule implementation as-is. Only cpp/game and the small set
# of core support files needed by it are compiled; KataGo's search, neural net,
# command-line, and data-I/O layers stay outside this project's build.
set(_go_katago_root "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/KataGo")
add_library(go_katago_game STATIC
    "${_go_katago_root}/cpp/core/bsearch.cpp"
    "${_go_katago_root}/cpp/core/global.cpp"
    "${_go_katago_root}/cpp/core/hash.cpp"
    "${_go_katago_root}/cpp/core/md5.cpp"
    "${_go_katago_root}/cpp/core/rand.cpp"
    "${_go_katago_root}/cpp/core/sha2.cpp"
    "${_go_katago_root}/cpp/core/test.cpp"
    "${_go_katago_root}/cpp/core/timer.cpp"
    "${_go_katago_root}/cpp/game/board.cpp"
    "${_go_katago_root}/cpp/game/boardhistory.cpp"
    "${_go_katago_root}/cpp/game/graphhash.cpp"
    "${_go_katago_root}/cpp/game/rules.cpp"
)
target_include_directories(go_katago_game PUBLIC "${_go_katago_root}/cpp")
target_compile_definitions(go_katago_game PUBLIC COMPILE_MAX_BOARD_LEN=19)
if(WIN32)
    target_compile_definitions(go_katago_game PRIVATE
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        LITTLE_ENDIAN=1234
        BIG_ENDIAN=4321
        BYTE_ORDER=LITTLE_ENDIAN
    )
    target_link_libraries(go_katago_game PUBLIC ws2_32)
endif()
set_target_properties(go_katago_game PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
unset(_go_katago_root)

add_library(go_rules STATIC src/go/game.cpp src/go/features.cpp)
target_include_directories(go_rules PUBLIC include)
target_link_libraries(go_rules PRIVATE go_katago_game)

add_executable(go_rules_cli src/go/rules_cli.cpp)
target_link_libraries(go_rules_cli PRIVATE go_rules go_katago_game)

add_library(go_nngames STATIC src/go/nngames.cpp)
target_link_libraries(go_nngames PUBLIC go_rules PRIVATE neuralnet::nn)

add_executable(go_data_cli src/go/data_cli.cpp)
target_link_libraries(go_data_cli PRIVATE go_nngames go_katago_game neuralnet::nn)

add_library(go_feature_cache STATIC src/go/feature_cache.cpp)
target_include_directories(go_feature_cache PUBLIC include)

add_executable(go_train_cli src/go/go_train_cli.cpp)
target_link_libraries(
    go_train_cli
    PRIVATE go_feature_cache go_katago_game neuralnet::nn
)
if(WIN32)
    # neuralnet.cpp's streaming SHA-256 uses a 1 MiB stack buffer. The Windows
    # default executable stack is also about 1 MiB, so model integrity checks
    # overflow before the loader can read the first parameter.
    target_link_options(go_train_cli PRIVATE "LINKER:/STACK:8388608")
endif()

# M6: the policy player wires a loaded strategy model to KataGo rules
# (features -> forward -> argmax over legal points). The GTP adapter and the
# arena both build on it so their move selection is identical by construction.
add_library(go_policy STATIC src/go/policy.cpp)
target_link_libraries(go_policy PUBLIC go_rules go_katago_game neuralnet::nn)

add_executable(go_gtp_cli src/go/go_gtp_cli.cpp)
target_link_libraries(go_gtp_cli PRIVATE go_policy)
if(WIN32)
    target_link_options(go_gtp_cli PRIVATE "LINKER:/STACK:8388608")
endif()

# M6: the arena plays batches of games between two engines (random / model /
# external GTP subprocess) and applies an SPRT win-rate test to the player.
add_executable(go_arena_cli src/go/go_arena_cli.cpp)
target_link_libraries(go_arena_cli PRIVATE go_policy)
if(WIN32)
    target_link_options(go_arena_cli PRIVATE "LINKER:/STACK:8388608")
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME go_neuralnet_m0_smoke COMMAND go_neuralnet_m0)
    set_tests_properties(go_neuralnet_m0_smoke PROPERTIES TIMEOUT 30)
    add_test(NAME go_train_cli_help COMMAND go_train_cli --help)
    set_tests_properties(go_train_cli_help PROPERTIES TIMEOUT 30)

    if(TARGET layer_bench)
        add_test(NAME go_neuralnet_m0_layer_bench_help COMMAND layer_bench --help)
        set_tests_properties(go_neuralnet_m0_layer_bench_help PROPERTIES TIMEOUT 30)
    endif()

    add_executable(go_rules_tests tests/go_rules_tests.cpp)
    target_link_libraries(go_rules_tests PRIVATE go_rules)
    add_test(NAME go_rules_m1_tests COMMAND go_rules_tests)
    set_tests_properties(go_rules_m1_tests PROPERTIES TIMEOUT 30)

    # Keep Python in the verification path without making it a build dependency.
    # Prefer the project's development venv, but allow an explicit interpreter.
    set(GO_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python interpreter for optional checks")
    if(NOT GO_PYTHON_EXECUTABLE)
        if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
            set(GO_PYTHON_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
            set(GO_PYTHON_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        else()
            find_package(Python3 COMPONENTS Interpreter QUIET)
            if(Python3_Interpreter_FOUND)
                set(GO_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
            endif()
        endif()
    endif()
    if(GO_PYTHON_EXECUTABLE)
        add_test(
            NAME go_rules_m1_sgf_python
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sgf_replay.py"
                    --cli "$<TARGET_FILE:go_rules_cli>"
        )
        set_tests_properties(go_rules_m1_sgf_python PROPERTIES
            TIMEOUT 30
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
        add_test(
            NAME go_pipeline_m2_m3_tests
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m2_m3.py"
        )
        set_tests_properties(go_pipeline_m2_m3_tests PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_data_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # M5: resume equivalence (2+2 epochs == 4 straight), dataset identity
        # gates, per-epoch metrics JSONL, and dual-role checkpoint artifacts.
        add_test(
            NAME go_trainer_m5_resume
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m5_resume.py"
        )
        set_tests_properties(go_trainer_m5_resume PROPERTIES
            TIMEOUT 300
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_train_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # M6: the C++ live feature extractor must match the Python trainer
        # bitwise, or GTP play would feed the network different inputs than
        # training saw.
        add_test(
            NAME go_features_m6_parity
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m6_features.py"
        )
        set_tests_properties(go_features_m6_parity PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_rules_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # M6: GTP adapter protocol behavior + arena engines + SPRT math.
        add_test(
            NAME go_gtp_arena_m6
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m6_gtp_arena.py"
        )
        set_tests_properties(go_gtp_arena_m6 PROPERTIES
            TIMEOUT 600
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_gtp_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # v2 cache: the per-sample move history must match the Python
        # tokenizer frame by frame, and a v1 cache must be refused rather than
        # silently substituted with zeros.
        add_test(
            NAME go_feature_cache_history
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_feature_cache_history.py"
        )
        set_tests_properties(go_feature_cache_history PROPERTIES
            TIMEOUT 120
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_train_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
    endif()
    find_program(GNUGO_EXECUTABLE gnugo)
    if(GO_PYTHON_EXECUTABLE AND GNUGO_EXECUTABLE)
        add_test(
            NAME go_rules_m1_gnugo_crosscheck
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gnugo_crosscheck.py"
                    --cli "$<TARGET_FILE:go_rules_cli>"
        )
        set_tests_properties(go_rules_m1_gnugo_crosscheck PROPERTIES TIMEOUT 30)
    endif()
endif()
