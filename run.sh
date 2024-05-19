#!/usr/bin/env bash

if [ ! -d build ]; then
  mkdir build
fi
pushd build > /dev/null
cmake .. && make -j8
echo

echo "=================================================="
echo ">> unittest: block encoders"
./bin/test_block_encoding 
echo
echo "=================================================="
echo ">> Contest Solution :)"
./bin/CCF_QDALS

echo
popd > /dev/null
