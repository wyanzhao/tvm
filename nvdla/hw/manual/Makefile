DEPTH := ../..

# the output dir
OUT_DIR := ${CURDIR}/../build/$(NVDLA_PROJECT)/manual
MODULES := NVDLA_CSC
MODULES += NVDLA_BDMA
MODULES += NVDLA_CACC
MODULES += NVDLA_CDMA
MODULES += NVDLA_CDP
MODULES += NVDLA_CDP_RDMA
MODULES += NVDLA_CMAC_A
MODULES += NVDLA_CMAC_B
MODULES += NVDLA_CVIF
MODULES += NVDLA_GEC
MODULES += NVDLA_GLB
MODULES += NVDLA_MCIF
MODULES += NVDLA_PDP
MODULES += NVDLA_PDP_RDMA
MODULES += NVDLA_RBK
MODULES += NVDLA_SDP
MODULES += NVDLA_SDP_RDMA
MODULES += nvdla
RDL_FILES := $(MODULES:%=%.rdl)

VCP_OUT_FILES := $(addprefix $(OUT_DIR)/,$(RDL_FILES))
# Make Rules
default: $(VCP_OUT_FILES) OUT_FILES 
	@echo "=============================================="
	@echo "files are generated under $(OUT_DIR)"
	@echo "=============================================="


VCP := $(CURDIR)/../tools/bin/vcp
PROJ_HEAD := $(CURDIR)/../build/${NVDLA_PROJECT}/spec/project.h
$(info PROJ_HEAD is $(PROJ_HEAD))

$(VCP_OUT_FILES) : $(OUT_DIR)/% : % $(PROJ_HEAD) $(VCP)
ifeq (,$(wildcard $(OUT_DIR)))
	@mkdir -p $(OUT_DIR)
endif
	$(VCP) -i $< -o $@ -imacros $(PROJ_HEAD) -cpp $(CPP)

.phony OUT_FILES: $(CURDIR)/RDLMerge.pl $(OUT_DIR)/nvdla.rdl $(RDL_FILES) Makefile
	$(AT)$(PERL) $(CURDIR)/RDLMerge.pl $(OUT_DIR)/nvdla.rdl $(JAVA) $(PERL) $(OUT_DIR)

# ======================
.PHONE: clean
clean:
	rm $(OUT_DIR) -rf
