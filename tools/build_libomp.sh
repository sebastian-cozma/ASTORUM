#!/usr/bin/env bash
# Build and install LLVM's OpenMP runtime (libomp) for macOS wheel builds.
#
# Why not Homebrew: brew bottles target the newest macOS (currently minos 26.0),
# so delocate refuses to vendor libomp.dylib into a wheel targeting macOS 11:
# "Library dependencies do not satisfy target MacOS version 11.0".
# See https://cibuildwheel.pypa.io/en/stable/faq/#macos-library-dependencies-do-not-satisfy-target-macos
# Building libomp from source with the wheel's MACOSX_DEPLOYMENT_TARGET
# produces a dylib that satisfies the check.
#
# Recipe: configure the openmp/ project standalone. It is NOT self-contained:
# it includes LLVMCheckCompilerLinkerFlag and ExtendPath from
# llvm-project/cmake/Modules, so that directory is extracted as well and put
# on CMAKE_MODULE_PATH. (Configuring the "runtimes" superbuild instead — as
# Homebrew does — fails with an empty PACKAGE_VERSION in
# get_clang_resource_dir and requires workarounds; the standalone path
# installs the classic layout: include/omp.h + lib/libomp.{dylib,so}.)
#
# Installs to ${LIBOMP_PREFIX} (default /tmp/libomp-install); point CMake's
# OpenMP_ROOT at the same path when building the extension.
#
# Used by cibuildwheel's before-all on macOS; also runs on Linux so the
# script itself can be tested locally. Idempotent: skips if already installed.

set -euo pipefail

LLVM_VERSION="18.1.8"
TARBALL_SHA256="0b58557a6d32ceee97c8d533a59b9212d87e0fc4d2833924eb6c611247db2f2a"
TARBALL="llvm-project-${LLVM_VERSION}.src.tar.xz"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${TARBALL}"
SRC_DIR="llvm-project-${LLVM_VERSION}.src"

PREFIX="${LIBOMP_PREFIX:-/tmp/libomp-install}"

if [ -e "${PREFIX}/lib/libomp.dylib" ] || [ -e "${PREFIX}/lib/libomp.so" ] || [ -e "${PREFIX}/lib/libomp.a" ]; then
    echo "[build_libomp] already installed at ${PREFIX}, skipping"
    exit 0
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "[build_libomp] downloading ${URL}"
curl -fsSL -o "${WORKDIR}/${TARBALL}" "${URL}"

echo "[build_libomp] verifying sha256"
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA="$(sha256sum "${WORKDIR}/${TARBALL}" | cut -d' ' -f1)"
else
    ACTUAL_SHA="$(shasum -a 256 "${WORKDIR}/${TARBALL}" | cut -d' ' -f1)"
fi
if [ "${ACTUAL_SHA}" != "${TARBALL_SHA256}" ]; then
    echo "[build_libomp] sha256 mismatch: expected ${TARBALL_SHA256}, got ${ACTUAL_SHA}" >&2
    exit 1
fi

echo "[build_libomp] extracting"
tar -xf "${WORKDIR}/${TARBALL}" -C "${WORKDIR}" \
    "${SRC_DIR}/openmp" \
    "${SRC_DIR}/cmake"

DEPLOYMENT_ARGS=()
if [ "$(uname)" = "Darwin" ]; then
    DEPLOYMENT_ARGS+=(
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}"
        # Absolute install name (Homebrew-style), not @rpath: delocate resolves
        # the extension's libomp reference through the dylib's install name,
        # and the extension carries no rpath entries to resolve @rpath with.
        "-DCMAKE_MACOSX_RPATH=OFF"
    )
fi

echo "[build_libomp] configuring (prefix: ${PREFIX})"
cmake -S "${WORKDIR}/${SRC_DIR}/openmp" -B "${WORKDIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_MODULE_PATH="${WORKDIR}/${SRC_DIR}/cmake/Modules" \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    "${DEPLOYMENT_ARGS[@]}"

echo "[build_libomp] building"
cmake --build "${WORKDIR}/build" --parallel

echo "[build_libomp] installing to ${PREFIX}"
cmake --install "${WORKDIR}/build"

if [ "$(uname)" = "Darwin" ]; then
    expected_id="${PREFIX}/lib/libomp.dylib"
    actual_id="$(otool -D "${expected_id}" | tail -n +2 | tr -d '[:space:]')"
    if [ "${actual_id}" != "${expected_id}" ]; then
        echo "[build_libomp] forcing install name: ${actual_id} -> ${expected_id}"
        install_name_tool -id "${expected_id}" "${expected_id}"
        # install_name_tool invalidates the ad-hoc code signature; arm64 macOS
        # refuses to load unsigned/invalidly-signed dylibs.
        codesign --force --sign - "${expected_id}"
    fi
fi

echo "[build_libomp] done"
