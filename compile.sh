#!/bin/zsh

mkdir -p build
cd build
echo "\n---- Cmake compilation ---- "
cmake ..
echo "\n---- Making Program ---- \n"
make
cd ..

