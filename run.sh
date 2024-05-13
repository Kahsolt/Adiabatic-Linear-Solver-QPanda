#!/usr/bin/env bash

if [ ! -d build ]; then
  mkdir build
fi
pushd build
cmake .. && make -j8
./bin/CCF_QDALS
popd > /dev/null
