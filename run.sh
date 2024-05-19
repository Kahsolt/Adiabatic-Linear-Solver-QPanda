#!/usr/bin/env bash

if [ ! -d build ]; then
  mkdir build
fi
pushd build > /dev/null
cmake .. && make -j8
echo
./bin/CCF_QDALS
popd > /dev/null
