# embed_text.cmake — 把文本文件嵌进生成的 C++ 头（raw string literal）。
#
# 用法：
#   cmake -DINPUT=<file> -DOUTPUT=<header> -DVAR=<identifier> -P embed_text.cmake
#
# 生成的头只含一个 inline constexpr char VAR[]，内容与 INPUT 逐字节一致。
# raw string 定界符取 )GPGUIMARK"（输入含该序列即报错）。
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED VAR)
    message(FATAL_ERROR "embed_text.cmake requires -DINPUT, -DOUTPUT and -DVAR")
endif()
file(READ "${INPUT}" _text)
string(FIND "${_text}" [==[)GPGUIMARK"]==] _delimiter_pos)
if(NOT _delimiter_pos EQUAL -1)
    message(FATAL_ERROR "${INPUT} contains the raw-string delimiter )GPGUIMARK\"")
endif()
string(CONCAT _out
    [==[// 由 cmake/embed_text.cmake 从 ]==]
    "${INPUT}"
    [==[ 生成，请勿手改。
#pragma once

namespace go::gui::page {

inline constexpr char ]==]
    "${VAR}"
    [==[[] = R"GPGUIMARK(]==]
    "${_text}"
    [==[)GPGUIMARK";

}  // namespace go::gui::page
]==])
file(WRITE "${OUTPUT}" "${_out}")
