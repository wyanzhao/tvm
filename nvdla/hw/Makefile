CC = gcc
 
ROOT := $(shell pwd)

# Process VTA JSON config
NVDLA_CONFIG := $(ROOT)/../config/nvdla_config.py
# Derive config name
NVDLA_PROJECT := $(shell python3 ${NVDLA_CONFIG} --target)
BUILD_DIR = $(ROOT)/build/$(NVDLA_PROJECT)

SYSTEMC := $(CURDIR)/../dep/systemc-2.3.0a/build
CPP := /usr/bin/cpp
JAVA := /usr/bin/java
PERL := /usr/bin/perl
CC := /usr/bin/gcc-4.8
CXX := /usr/bin/g++-4.8

TARGET ?= libnvdla_cmod.so

default: $(BUILD_DIR)/$(TARGET)
	@echo "=============================================="
	@echo "files are generated under $(BUILD_DIR)"
	@echo "=============================================="

$(BUILD_DIR)/$(TARGET): manual
	$(MAKE) -C $(ROOT)/cmod NVDLA_PROJECT=$(NVDLA_PROJECT) CC=$(CC) CXX=$(CXX) SYSTEMC=$(SYSTEMC)

manual: spec
	$(MAKE) -C $(ROOT)/manual NVDLA_PROJECT=$(NVDLA_PROJECT) CPP=$(CPP) JAVA=$(JAVA) PERL=$(PERL)
  
spec:
	mkdir -p $(BUILD_DIR)/spec && cd $(BUILD_DIR)/spec && python3 ${NVDLA_CONFIG} --gen-headers

.PHONY: clean
	
clean:
	-rm -rf ${ROOT}/build