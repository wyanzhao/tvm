# NVDLA Library for TVM

## Environment requirement

OS: Ubuntu18.04

Toolchain: 

`sudo apt install build-essential`

`sudo apt install automake`

## Build protobuf library

* `cd nvdla/external/protobuf-2.6`

* `mkdir -p build`

* `cd build`

* `../configure --prefix=${HOME}/.local`

* `make -j$(nproc)`

* `make install`

After those steps, you can check if in your `${HOME}/.local/lib` directory has protobuf libs.

## Build nvdla library

`cd nvdla/`

`make compiler -j$(nproc)`

After this process, `libnvdla_compiler.so` should be in your nvdla/out directory.
