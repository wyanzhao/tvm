<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/incubator-tvm-site/master/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://docs.tvm.ai) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tvm.ai/buildStatus/icon?job=tvm/master)](https://ci.tvm.ai/job/tvm/job/master/)
[![Azure Pipeline](https://dev.azure.com/tvmai/tvm/_apis/build/status/windows_mac_build?branchName=master)](https://dev.azure.com/tvmai/tvm/_build/latest?definitionId=2&branchName=master)

Apache TVM (incubating) is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Checkout the [Contributor Guide](https://docs.tvm.ai/contribute/)

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): TVM uses [HalideIR](https://github.com/dmlc/HalideIR) as data structure for
  arithmetic simplification and low level lowering. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.

TVM for NVDLA build instruction
---------------

`git clone git@github.com:wyanzhao/tvm.git --recursive`

First step, follow those steps to build NVDLA related libraries
- `cd tvm/nvdla`
- [NVDLA README](https://github.com/wyanzhao/nvdla)

Then get ready to build TVM for NVDLA
- `sudo apt-get update`

- `sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-9`

- `cd tvm`

- `mkdir -p build`

- `cp cmake/config.cmake build`

- TVM optionally depends on LLVM. LLVM is required for CPU and NVDLA codegen that needs LLVM.
    - Edit ``build/config.cmake`` to customize the compilation options
    - Note that apt-package append ``llvm-config`` with version number.
      For example, set ``set(LLVM_CONFIG llvm-config-9)`` if you installed 4.0 package
      
 - `cd build`
 
 - `cmake ..`
 
 - `make -j${nproc}`
 
Python Package Installation
---------------------------

The python package is located at `tvm/python`

   This method is **recommended for developers** who may change the codes.

   Set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `tvm` on the home directory
   `~`. then we can added the following line in `~/.bashrc`.
   The changes will be immediately reflected once you pull the code and rebuild the project (no need to call ``setup`` again)

       export TVM_HOME=/path/to/tvm
       export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nvdla/python:${PYTHONPATH}
 
 Python dependencies
---------------------------
   * Necessary dependencies:

       `pip3 install --user numpy decorator attrs`
       
 Build test
---------------------------
  - In Visual Studio Code, if every steps succeed, run `turotials/frontend/nvdla_uinit_test/lenet_test_without_softmax.py` is able to build nvdla loadable.
