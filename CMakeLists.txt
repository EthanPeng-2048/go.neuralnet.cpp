cmake_minimum_required(VERSION 3.30)

project(go_neuralnet
    VERSION 0.1.0
    DESCRIPTION "19x19 Go training and inference built on neuralnet.cpp"
    HOMEPAGE_URL "https://github.com/EthanPeng-2048/go.neuralnet.cpp"
    LANGUAGES CXX
)

# C++26. CMake 4.x has no CXX26 dialect flag for MSVC (cl 19.51 routes C++26
# features through /std:c++latest), so mirror the library's own split: MSVC gets
# the C++23 dialect plus an explicit /std:c++latest, everyone else gets C++26.
if(MSVC)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(/std:c++latest)
else()
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

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
# 并行重放校验（verify_replay 的 --threads）用 std::thread。
find_package(Threads REQUIRED)
target_link_libraries(go_nngames PUBLIC go_rules Threads::Threads PRIVATE neuralnet::nn)

add_executable(go_data_cli src/go/data_cli.cpp)
target_link_libraries(go_data_cli PRIVATE go_nngames go_katago_game neuralnet::nn)

add_library(go_feature_cache STATIC src/go/feature_cache.cpp)
target_include_directories(go_feature_cache PUBLIC include)

# M7: the B architecture (dual-branch encoder + decision query + policy/value
# heads) in its own library so trainer and policy player share one model
# factory/spec dispatch. Element-wise math is written with dsl::compute; on GPU
# builds each executable additionally compiles the model TU into its own
# AOT fusion collector (see nn_enable_gpu_fusion below).
add_library(go_model STATIC src/go/go_model.cpp)
target_include_directories(go_model PUBLIC include)
target_link_libraries(go_model PUBLIC neuralnet::nn go_feature_cache)

add_executable(go_train_cli src/go/go_train_cli.cpp)
target_link_libraries(
    go_train_cli
    PRIVATE go_model go_feature_cache go_katago_game neuralnet::nn
)
if(WIN32)
    # neuralnet.cpp's streaming SHA-256 uses a 1 MiB stack buffer. The Windows
    # default executable stack is also about 1 MiB, so model integrity checks
    # overflow before the loader can read the first parameter.
    target_link_options(go_train_cli PRIVATE "LINKER:/STACK:8388608")
endif()

# M7: AOT fusion registration (neuralnet.cpp 的库外使用者标准做法）。
# `nn_enable_gpu_fusion` 只为 Vulkan 构建定义；它把**目标自己的 SOURCES**
# 减去 MAIN 后以 -DNN_EXPR_SCAN 编进收集器，GoModel 里每个 dsl::compute 的
# FusedAnchor 在静态初始化期自登记。模型 TU 因此必须出现在目标的源文件
# 列表里（它同时也在 go_model 静态库里——链接器不会重复取归档成员）。
if(TARGET compile_gpu_shaders)
    target_sources(go_train_cli PRIVATE src/go/go_model.cpp)
    nn_enable_gpu_fusion(go_train_cli MAIN src/go/go_train_cli.cpp)
endif()

# M6: the policy player wires a loaded strategy model to KataGo rules
# (features -> forward -> argmax over legal points). The GTP adapter and the
# arena both build on it so their move selection is identical by construction.
add_library(go_policy STATIC src/go/policy.cpp)
target_link_libraries(go_policy PUBLIC go_rules go_katago_game go_model neuralnet::nn)

# M8: MCTS inference engine (PUCT / prior normalization / legality /
# transposition table), search-only per D9. Model-agnostic: priors and leaf
# values come in through go::LeafEvaluator; the product line plugs in
# PolicyPlayer via make_policy_evaluator.
add_library(go_mcts STATIC src/go/mcts.cpp)
target_link_libraries(go_mcts PUBLIC go_policy)

add_executable(go_gtp_cli src/go/go_gtp_cli.cpp)
target_link_libraries(go_gtp_cli PRIVATE go_policy go_mcts)
if(WIN32)
    target_link_options(go_gtp_cli PRIVATE "LINKER:/STACK:8388608")
endif()

# M6: the arena plays batches of games between two engines (random / model /
# external GTP subprocess) and applies an SPRT win-rate test to the player.
add_executable(go_arena_cli src/go/go_arena_cli.cpp)
target_link_libraries(go_arena_cli PRIVATE go_arena_engine go_policy)
if(WIN32)
    target_link_options(go_arena_cli PRIVATE "LINKER:/STACK:8388608")
endif()

# 推理侧同样需要模型 TU 进各自目标的融合收集器。
if(TARGET compile_gpu_shaders)
    target_sources(go_gtp_cli PRIVATE src/go/go_model.cpp)
    nn_enable_gpu_fusion(go_gtp_cli MAIN src/go/go_gtp_cli.cpp)
    target_sources(go_arena_cli PRIVATE src/go/go_model.cpp)
    nn_enable_gpu_fusion(go_arena_cli MAIN src/go/go_arena_cli.cpp)
endif()

# GUI: 对局引擎抽象（random / model / 外部 GTP 子进程）抽出共用，
# go_arena_cli 的批量评估与 GUI 竞技台走同一实现。
add_library(go_arena_engine STATIC src/go/arena_engine.cpp)
target_link_libraries(go_arena_engine PUBLIC go_policy go_mcts)

# GUI: 会话状态机（对弈 + 竞技台）+ 极简 HTTP 服务器 + 内嵌单页前端。
# 规则固定中国规则（go::RuleSet::chinese），选点走 go::PolicyPlayer。
find_package(Threads REQUIRED)
add_library(go_gui STATIC src/go/gui_session.cpp src/go/http_server.cpp)
target_include_directories(go_gui PUBLIC include)
target_link_libraries(go_gui PUBLIC go_arena_engine go_policy Threads::Threads)

# gui/index.html 在构建期嵌入 go_gui_cli（raw string literal）。
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gui_page.hpp"
    COMMAND ${CMAKE_COMMAND}
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/gui/index.html"
        "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/gui_page.hpp"
        "-DVAR=kGuiPage"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS gui/index.html cmake/embed_text.cmake
    COMMENT "Embedding gui/index.html into gui_page.hpp"
    VERBATIM)

add_executable(go_gui_cli
    src/go/go_gui_cli.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/gui_page.hpp")
target_include_directories(go_gui_cli PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(go_gui_cli PRIVATE go_gui)
if(WIN32)
    target_link_options(go_gui_cli PRIVATE "LINKER:/STACK:8388608")
endif()

if(TARGET compile_gpu_shaders)
    target_sources(go_gui_cli PRIVATE src/go/go_model.cpp)
    nn_enable_gpu_fusion(go_gui_cli MAIN src/go/go_gui_cli.cpp)
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

    # M7: GoModel 契约（常量 / ModelSpec 往返 / tiny forward-backward / GoLoss）。
    add_executable(go_model_tests tests/go_model_tests.cpp)
    target_link_libraries(go_model_tests PRIVATE go_model)
    add_test(NAME go_model_m7_tests COMMAND go_model_tests)
    set_tests_properties(go_model_m7_tests PROPERTIES TIMEOUT 120)

    # M8: MCTS engine contracts (PUCT selection / prior renormalization /
    # legality / transposition table / terminal scoring).
    add_executable(go_mcts_tests tests/go_mcts_tests.cpp)
    target_link_libraries(go_mcts_tests PRIVATE go_mcts)
    add_test(NAME go_mcts_m8_tests COMMAND go_mcts_tests)
    set_tests_properties(go_mcts_m8_tests PROPERTIES TIMEOUT 120)

    # GUI: 会话状态机单测（对弈流程、悔棋、竞技台步进与换先、SGF、状态 JSON）。
    add_executable(go_gui_tests tests/go_gui_tests.cpp)
    target_link_libraries(go_gui_tests PRIVATE go_gui)
    add_test(NAME go_gui_session_tests COMMAND go_gui_tests)
    set_tests_properties(go_gui_session_tests PROPERTIES TIMEOUT 120)

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
            TIMEOUT 900
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
        # M7: dual-head metrics + value targets must not be degenerate, B
        # resume equivalence, and GTP genmove legality with a B model. Runs a
        # tiny B config; the full B-vs-C claim lives in docs/benchmarks/m7-b-vs-c.md.
        add_test(
            NAME go_model_m7_python
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m7_b_model.py"
        )
        set_tests_properties(go_model_m7_python PROPERTIES
            TIMEOUT 600
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_train_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # M8: the MCTS engine behind the GTP (--visits) and arena
        # (mcts:VISITS:model:...) front ends. Tree semantics live in
        # go_mcts_tests; this covers the product wiring.
        add_test(
            NAME go_mcts_m8_python
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_m8_mcts.py"
        )
        set_tests_properties(go_mcts_m8_python PROPERTIES
            TIMEOUT 600
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_gtp_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
        # GUI: 起一个真实 go_gui_cli，走 HTTP 全流程（页面、对弈、引擎应手、
        # 悔棋、竞技台自动对局、SGF、评估接口）。
        add_test(
            NAME go_gui_http
            COMMAND "${GO_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gui_http.py"
        )
        set_tests_properties(go_gui_http PROPERTIES
            TIMEOUT 300
            ENVIRONMENT "GO_TEST_BINARY_DIR=$<TARGET_FILE_DIR:go_gui_cli>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
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
