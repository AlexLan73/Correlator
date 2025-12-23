#!/bin/bash
set -e

echo "=== FFT Correlator Debug Build ==="
echo

# Очистка предыдущей отладочной сборки
echo "Cleaning previous debug build..."
rm -rf build_debug

# Конфигурация CMake для Debug
echo "Configuring CMake for Debug build..."
mkdir -p build_debug
cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Сборка
echo "Building with debug symbols..."
make -j$(nproc)

echo
echo "=== Debug Build Complete ==="
echo "Executable: $(pwd)/Correlator"
echo "Debug symbols: included"
echo
echo "To run with gdb:"
echo "  gdb ./Correlator"
echo
echo "To run directly:"
echo "  ./Correlator"
echo
echo "To debug in VS Code:"
echo "  Press F5 and select 'Launch Correlator (Debug)'"

