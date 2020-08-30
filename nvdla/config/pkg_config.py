# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""VTA Package configuration module

This module is dependency free and can be used to configure package.
"""
from __future__ import absolute_import as _abs

import json
import glob
import math

class PkgConfig(object):
    """Simple package config tool for VTA.

    This is used to provide runtime specific configurations.

    Parameters
    ----------
    cfg : dict
        The config dictionary

    proj_root : str
        Path to the project root
    """
    cfg_keys = [
  "FEATURE_DATA_TYPE_INT8",
  "WEIGHT_DATA_TYPE_INT8",
  "WEIGHT_COMPRESSION_ENABLE",
  "WINOGRAD_ENABLE",
  "BATCH_ENABLE",
  "SECONDARY_MEMIF_ENABLE",
  "SDP_LUT_ENABLE",
  "SDP_BS_ENABLE",
  "SDP_BN_ENABLE",
  "SDP_EW_ENABLE",
  "BDMA_ENABLE",
  "RUBIK_ENABLE",
  "RUBIK_CONTRACT_ENABLE",
  "RUBIK_RESHAPE_ENABLE",
  "PDP_ENABLE",
  "CDP_ENABLE",
  "RETIMING_ENABLE",
  "MAC_ATOMIC_C_SIZE",
  "MAC_ATOMIC_K_SIZE",
  "MEMORY_ATOMIC_SIZE",
  "MAX_BATCH_SIZE",
  "CBUF_BANK_NUMBER",
  "CBUF_BANK_WIDTH",
  "CBUF_BANK_DEPTH",
  "SDP_BS_THROUGHPUT",
  "SDP_BN_THROUGHPUT",
  "SDP_EW_THROUGHPUT",
  "PDP_THROUGHPUT",
  "CDP_THROUGHPUT",
  "PRIMARY_MEMIF_LATENCY",
  "SECONDARY_MEMIF_LATENCY",
  "PRIMARY_MEMIF_MAX_BURST_LENGTH",
  "PRIMARY_MEMIF_WIDTH",
  "SECONDARY_MEMIF_MAX_BURST_LENGTH",
  "SECONDARY_MEMIF_WIDTH",
  "MEM_ADDRESS_WIDTH",
  "NUM_DMA_READ_CLIENTS",
  "NUM_DMA_WRITE_CLIENTS",
  "target"
    ]

    def __init__(self, cfg, proj_root):

        # Update cfg now that we've extended it
        self.__dict__.update(cfg)
        

        # Derived parameters
        if cfg.get('FEATURE_DATA_TYPE_INT8', False) ==  True:
            cfg['FEATURE_DATA_TYPE_INT8'] = 1
            cfg['BPE'] = 8
        else:
            raise ValueError("NVDLA_FEATURE_DATA_TYPE_INT8 must be set")

        if cfg.get('WEIGHT_DATA_TYPE_INT8', False) ==  True:
            cfg['WEIGHT_DATA_TYPE_INT8'] = 1
        else:
            raise ValueError("NVDLA_WEIGHT_DATA_TYPE_INT8 must be set")

        if cfg.get('WEIGHT_COMPRESSION_ENABLE', None) ==  True:
            cfg['WEIGHT_COMPRESSION_ENABLE'] = 1
        elif cfg.get('WEIGHT_COMPRESSION_ENABLE', None) ==  False:
            del cfg['WEIGHT_COMPRESSION_ENABLE']
            cfg['WEIGHT_COMPRESSION_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_WEIGHT_COMPRESSION_{EN,DIS}ABLE must be set")

        if cfg.get('WINOGRAD_ENABLE', None) ==  True:
            cfg['WINOGRAD_ENABLE'] = 1
        elif cfg.get('WINOGRAD_ENABLE', None) ==  False:
            del cfg['WINOGRAD_ENABLE']
            cfg['WINOGRAD_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_WINOGRAD_{EN,DIS}ABLE must be set")

        if cfg.get('BATCH_ENABLE', None) ==  True:
            cfg['BATCH_ENABLE'] = 1
        elif cfg.get('BATCH_ENABLE', None) ==  False:
            del cfg['BATCH_ENABLE']
            cfg['BATCH_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_BATCH_{EN,DIS}ABLE must be set")
        
        if cfg.get('SECONDARY_MEMIF_ENABLE', None) ==  True:
            cfg['SECONDARY_MEMIF_ENABLE'] = 1
        elif cfg.get('SECONDARY_MEMIF_ENABLE', None) ==  False:
            del cfg['SECONDARY_MEMIF_ENABLE']
            cfg['SECONDARY_MEMIF_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_SECONDARY_MEMIF_{EN,DIS}ABLE must be set")

        if cfg.get('SDP_LUT_ENABLE', None) ==  True:
            cfg['SDP_LUT_ENABLE'] = 1
        elif cfg.get('SDP_LUT_ENABLE', None) ==  False:
            del cfg['SDP_LUT_ENABLE']
            cfg['SDP_LUT_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_SDP_LUT_{EN,DIS}ABLE must be set")
        
        if cfg.get('SDP_BS_ENABLE', None) ==  True:
            cfg['SDP_BS_ENABLE'] = 1
        elif cfg.get('SDP_BS_ENABLE', None) ==  False:
            del cfg['SDP_BS_ENABLE']
            cfg['SDP_BS_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_SDP_BS_{EN,DIS}ABLE must be set")
        
        if cfg.get('SDP_BN_ENABLE', None) ==  True:
            cfg['SDP_BN_ENABLE'] = 1
        elif cfg.get('SDP_BN_ENABLE', None) ==  False:
            del cfg['SDP_BN_ENABLE']
            cfg['SDP_BN_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_SDP_BN_{EN,DIS}ABLE must be set")

            
        if cfg.get('SDP_EW_ENABLE', None) ==  True:
            cfg['SDP_EW_ENABLE'] = 1
        elif cfg.get('SDP_EW_ENABLE', None) ==  False:
            del cfg['SDP_EW_ENABLE']
            cfg['SDP_EW_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_SDP_EW_{EN,DIS}ABLE must be set")
        
        if cfg.get('BDMA_ENABLE', None) ==  True:
            cfg['BDMA_ENABLE'] = 1
        elif cfg.get('BDMA_ENABLE', None) ==  False:
            del cfg['BDMA_ENABLE']
            cfg['BDMA_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_BDMA_{EN,DIS}ABLE must be set")
        
        if cfg.get('RUBIK_ENABLE', None) ==  True:
            cfg['RUBIK_ENABLE'] = 1
        elif cfg.get('RUBIK_ENABLE', None) ==  False:
            del cfg['RUBIK_ENABLE']
            cfg['RUBIK_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_RUBIK_{EN,DIS}ABLE must be set")

        if cfg.get('RUBIK_CONTRACT_ENABLE', None) ==  True:
            cfg['RUBIK_CONTRACT_ENABLE'] = 1
        elif cfg.get('RUBIK_CONTRACT_ENABLE', None) ==  False:
            del cfg['RUBIK_CONTRACT_ENABLE']
            cfg['RUBIK_CONTRACT_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_RUBIK_CONTRACT_{EN,DIS}ABLE must be set")
        
        if cfg.get('RUBIK_RESHAPE_ENABLE', None) ==  True:
            cfg['RUBIK_RESHAPE_ENABLE'] = 1
        elif cfg.get('RUBIK_RESHAPE_ENABLE', None) ==  False:
            del cfg['RUBIK_RESHAPE_ENABLE']
            cfg['RUBIK_RESHAPE_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_RUBIK_RESHAPE_{EN,DIS}ABLE must be set")

        if cfg.get('PDP_ENABLE', None) ==  True:
            cfg['PDP_ENABLE'] = 1
        elif cfg.get('PDP_ENABLE', None) ==  False:
            del cfg['PDP_ENABLE']
            cfg['PDP_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_PDP_{EN,DIS}ABLE must be set")

        if cfg.get('CDP_ENABLE', None) ==  True:
            cfg['CDP_ENABLE'] = 1
        elif cfg.get('CDP_ENABLE', None) ==  False:
            del cfg['CDP_ENABLE']
            cfg['CDP_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_CDP_{EN,DIS}ABLE must be set")
        
        if cfg.get('RETIMING_ENABLE', None) ==  True:
            cfg['RETIMING_ENABLE'] = 1
        elif cfg.get('RETIMING_ENABLE', None) ==  False:
            del cfg['RETIMING_ENABLE']
            cfg['RETIMING_DISABLE'] = 1
        else:
            raise ValueError("one of NVDLA_RETIMING_{EN,DIS}ABLE must be set")
        
        if cfg.get('MAC_ATOMIC_C_SIZE', None) != None:
            assert cfg['MAC_ATOMIC_C_SIZE'] == 64 or \
            cfg['MAC_ATOMIC_C_SIZE'] == 32 or \
            cfg['MAC_ATOMIC_C_SIZE'] == 8
        else:
            raise ValueError("one of NVDLA_MAC_ATOMIC_C_SIZE_{64,32,8} must be set")
        
        if cfg.get('MAC_ATOMIC_K_SIZE', None) != None:
            assert cfg['MAC_ATOMIC_K_SIZE'] == 32 or \
            cfg['MAC_ATOMIC_K_SIZE'] == 16 or \
            cfg['MAC_ATOMIC_K_SIZE'] == 8
        else:
            raise ValueError("one of NVDLA_MAC_ATOMIC_K_SIZE_{32,16,8} must be set")
        
        if cfg.get('MEMORY_ATOMIC_SIZE', None) != None:
            assert cfg['MEMORY_ATOMIC_SIZE'] == 32 or \
            cfg['MEMORY_ATOMIC_SIZE'] == 16 or \
            cfg['MEMORY_ATOMIC_SIZE'] == 8
        else:
            raise ValueError("one of NVDLA_MEMORY_ATOMIC_SIZE_{32,16,8} must be set")

        if cfg.get('MAX_BATCH_SIZE', None) != None:
            assert cfg['MAX_BATCH_SIZE'] == 32 or \
            cfg['MAX_BATCH_SIZE'] == 0

            if(cfg['MAX_BATCH_SIZE'] == 0):
                del cfg['MAX_BATCH_SIZE']
        else:
            raise ValueError("one of NVDLA_MAX_BATCH_SIZE_{32,x} must be set")

        if cfg.get('CBUF_BANK_NUMBER', None) != None:
            assert cfg['CBUF_BANK_NUMBER'] == 32 or \
            cfg['CBUF_BANK_NUMBER'] == 16
        else:
            raise ValueError("one of NVDLA_CBUF_BANK_NUMBER_{16,32} must be set")

        if cfg.get('CBUF_BANK_WIDTH', None) != None:
            assert cfg['CBUF_BANK_WIDTH'] == 32 or \
            cfg['CBUF_BANK_WIDTH'] == 8 or \
            cfg['CBUF_BANK_WIDTH'] == 64
        else:
            raise ValueError("one of NVDLA_CBUF_BANK_WIDTH_{64,32,8} must be set")

        if cfg.get('CBUF_BANK_DEPTH', None) != None:
            assert cfg['CBUF_BANK_DEPTH'] == 512 or \
            cfg['CBUF_BANK_DEPTH'] == 128
        else:
            raise ValueError("only NVDLA_CBUF_BANK_DEPTH_{512,128} can be set")

        if cfg.get("SDP_BS_ENABLE", None) != None:
            if cfg.get('SDP_BS_THROUGHPUT', None) != None:
                assert cfg['SDP_BS_THROUGHPUT'] == 16 or \
                cfg['SDP_BS_THROUGHPUT'] == 8 or \
                cfg['SDP_BS_THROUGHPUT'] == 4 or \
                cfg['SDP_BS_THROUGHPUT'] == 2 or \
                cfg['SDP_BS_THROUGHPUT'] == 1
            else:
                raise ValueError("one of NVDLA_SDP_BS_THROUGHPUT_{16,8,4,2,1} must be set")
        else:
            cfg['SDP_BS_THROUGHPUT'] = 0

        if cfg.get("SDP_BN_ENABLE", None) != None:
            if cfg.get('SDP_BN_THROUGHPUT', None) != None:
                assert cfg['SDP_BN_THROUGHPUT'] == 16 or \
                cfg['SDP_BN_THROUGHPUT'] == 8 or \
                cfg['SDP_BN_THROUGHPUT'] == 4 or \
                cfg['SDP_BN_THROUGHPUT'] == 2 or \
                cfg['SDP_BN_THROUGHPUT'] == 1
            else:
                raise ValueError("one of NVDLA_SDP_BN_THROUGHPUT_{16,8,4,2,1} must be set")
        else:
            cfg['SDP_BN_THROUGHPUT'] = 0


        if cfg.get("SDP_EW_ENABLE", None) != None:
            if cfg.get('SDP_EW_THROUGHPUT', None) != None:
                assert cfg['SDP_EW_THROUGHPUT'] == 4 or \
                cfg['SDP_EW_THROUGHPUT'] == 2 or \
                cfg['SDP_EW_THROUGHPUT'] == 1 or \
                cfg['SDP_EW_THROUGHPUT'] == 0

                if cfg['SDP_EW_THROUGHPUT'] == 0:
                    cfg['SDP_EW_THROUGHPUT'] = 1
            else:
                raise ValueError("one of NVDLA_SDP_EW_THROUGHPUT_{4,2,1,x} must be set")
            cfg['SDP_EW_THROUGHPUT_LOG2'] = int(math.log2(cfg['SDP_EW_THROUGHPUT']))
        else:
            cfg['SDP_EW_THROUGHPUT'] = 0
            cfg['SDP_EW_THROUGHPUT_LOG2'] = 0

        cfg['SDP_MAX_THROUGHPUT'] = max(cfg['SDP_EW_THROUGHPUT'], max(cfg['SDP_BN_THROUGHPUT'], cfg['SDP_BS_THROUGHPUT']))
        cfg['SDP2PDP_WIDTH'] = cfg['SDP_MAX_THROUGHPUT'] * cfg['BPE']

        if cfg.get('PDP_THROUGHPUT', None) != None:
            assert cfg['PDP_THROUGHPUT'] == 8 or \
            cfg['PDP_THROUGHPUT'] == 4 or \
            cfg['PDP_THROUGHPUT'] == 2 or \
            cfg['PDP_THROUGHPUT'] == 1
        else:
            raise ValueError("one of NVDLA_PDP_THROUGHPUT_{8,4,2,1} must be set")

        if cfg.get('PDP_THROUGHPUT', None) != None:
            assert cfg['PDP_THROUGHPUT'] == 8 or \
            cfg['PDP_THROUGHPUT'] == 4 or \
            cfg['PDP_THROUGHPUT'] == 2 or \
            cfg['PDP_THROUGHPUT'] == 1
        else:
            raise ValueError("one of NVDLA_PDP_THROUGHPUT_{8,4,2,1} must be set")

        if cfg.get('CDP_THROUGHPUT', None) != None:
            assert cfg['CDP_THROUGHPUT'] == 8 or \
            cfg['CDP_THROUGHPUT'] == 4 or \
            cfg['CDP_THROUGHPUT'] == 2 or \
            cfg['CDP_THROUGHPUT'] == 1
        else:
            raise ValueError("one of NVDLA_CDP_THROUGHPUT_{8,4,2,1} must be set")

        if cfg.get('PRIMARY_MEMIF_LATENCY', None) != None:
            assert cfg['PRIMARY_MEMIF_LATENCY'] == 1024 or \
            cfg['PRIMARY_MEMIF_LATENCY'] == 256 or \
            cfg['PRIMARY_MEMIF_LATENCY'] == 64
        else:
            raise ValueError("one of NVDLA_PRIMARY_MEMIF_LATENCY_{1024,256,64} must be set")

        if cfg.get('SECONDARY_MEMIF_LATENCY', None) != None:
            assert cfg['SECONDARY_MEMIF_LATENCY'] == 1024 or \
            cfg['SECONDARY_MEMIF_LATENCY'] == 256 or \
            cfg['SECONDARY_MEMIF_LATENCY'] == 64 or \
            cfg['SECONDARY_MEMIF_LATENCY'] == 0 

            if cfg['SECONDARY_MEMIF_LATENCY'] == 0:
                del cfg['SECONDARY_MEMIF_LATENCY']
        else:
            raise ValueError("one of NVDLA_PRIMARY_MEMIF_LATENCY_{1024,256,64} must be set")

        if cfg.get('PRIMARY_MEMIF_MAX_BURST_LENGTH', None) != None:
            assert cfg['PRIMARY_MEMIF_MAX_BURST_LENGTH'] == 1 or \
            cfg['PRIMARY_MEMIF_MAX_BURST_LENGTH'] == 4
        else:
            raise ValueError("one of NVDLA_PRIMARY_MEMIF_MAX_BURST_LENGTH_{1,4} must be set")

        if cfg.get('PRIMARY_MEMIF_WIDTH', None) != None:
            assert cfg['PRIMARY_MEMIF_WIDTH'] == 256 or \
            cfg['PRIMARY_MEMIF_WIDTH'] == 128 or \
            cfg['PRIMARY_MEMIF_WIDTH'] == 64
        else:
            raise ValueError("one of NVDLA_PRIMARY_MEMIF_WIDTH_{256,128,64} must be set")

        if cfg.get('SECONDARY_MEMIF_MAX_BURST_LENGTH', None) != None:
            assert cfg['SECONDARY_MEMIF_MAX_BURST_LENGTH'] == 1 or \
            cfg['SECONDARY_MEMIF_MAX_BURST_LENGTH'] == 4 or \
            cfg['SECONDARY_MEMIF_MAX_BURST_LENGTH'] == 0

            if cfg['SECONDARY_MEMIF_MAX_BURST_LENGTH'] == 0:
                del cfg['SECONDARY_MEMIF_MAX_BURST_LENGTH']
        else:
            raise ValueError("one of NVDLA_SECONDARY_MEMIF_MAX_BURST_LENGTH_{1,4,x} must be set")

        if cfg.get('SECONDARY_MEMIF_WIDTH', None) != None:
            assert cfg['SECONDARY_MEMIF_WIDTH'] == 256 or \
            cfg['SECONDARY_MEMIF_WIDTH'] == 128 or \
            cfg['SECONDARY_MEMIF_WIDTH'] == 64 or \
            cfg['SECONDARY_MEMIF_WIDTH'] == 0

            if cfg['SECONDARY_MEMIF_WIDTH'] == 0:
                del cfg['SECONDARY_MEMIF_WIDTH']
        else:
            raise ValueError("one of NVDLA_SECONDARY_MEMIF_WIDTH_{256,128,64,x} must be set")

        if cfg.get('MEM_ADDRESS_WIDTH', None) != None:
            assert cfg['MEM_ADDRESS_WIDTH'] == 64 or \
            cfg['MEM_ADDRESS_WIDTH'] == 32
        else:
            raise ValueError("one of NVDLA_PRIMARY_MEMIF_WIDTH_{64,32} must be set")

        if cfg.get('SECONDARY_MEMIF_ENABLE', None) ==  True:
            cfg['MEMIF_WIDTH'] = max(cfg['PRIMARY_MEMIF_WIDTH'], max(cfg['SECONDARY_MEMIF_WIDTH'], cfg['MEMORY_ATOMIC_SIZE'] * cfg['BPE']))
        elif cfg.get('SECONDARY_MEMIF_DISABLE', None) ==  True:
            cfg['MEMIF_WIDTH'] = max(cfg['PRIMARY_MEMIF_WIDTH'], cfg['MEMORY_ATOMIC_SIZE'] * cfg['BPE'])
        else:
            raise ValueError("one of NVDLA_SECONDARY_MEMIF_{EN,DIS}ABLE must be set")

        cfg['DMA_RD_SIZE'] = 15
        cfg['DMA_WR_SIZE'] = 13
        cfg['DMA_MASK_BIT'] = int(cfg['MEMIF_WIDTH'] / cfg['BPE'] / cfg['MEMORY_ATOMIC_SIZE'])
        cfg['DMA_RD_RSP'] = int(cfg['MEMIF_WIDTH'] + cfg['DMA_MASK_BIT'])
        cfg['DMA_WR_REQ'] = int(cfg['MEMIF_WIDTH'] + cfg['DMA_MASK_BIT'] + 1)
        cfg['DMA_WR_CMD'] = int(cfg['MEM_ADDRESS_WIDTH'] + cfg['DMA_WR_SIZE'] + 1)
        cfg['DMA_RD_REQ'] = int(cfg['MEM_ADDRESS_WIDTH'] + cfg['DMA_RD_SIZE'])


        cfg['MEMORY_ATOMIC_LOG2'] = int(math.log2(cfg['MEMORY_ATOMIC_SIZE']))
        cfg['PRIMARY_MEMIF_WIDTH_LOG2'] = int(math.log2(cfg['PRIMARY_MEMIF_WIDTH'] / 8))
        
        
        if cfg.get('SECONDARY_MEMIF_WIDTH', None) != None:
            cfg['SECONDARY_MEMIF_WIDTH_LOG2'] = int(math.log2(cfg['SECONDARY_MEMIF_WIDTH'] / 8))
        
        cfg['MEMORY_ATOMIC_WIDTH'] = cfg['MEMORY_ATOMIC_SIZE'] * cfg['BPE']
        cfg['MCIF_BURST_SIZE'] = cfg['PRIMARY_MEMIF_MAX_BURST_LENGTH'] * cfg['DMA_MASK_BIT']
        cfg['MCIF_BURST_SIZE_LOG2'] = int(math.log2(cfg['MCIF_BURST_SIZE']))

        if cfg.get('NUM_DMA_READ_CLIENTS', None) != None:
            assert cfg['NUM_DMA_READ_CLIENTS'] == 10 or \
            cfg['NUM_DMA_READ_CLIENTS'] == 8 or \
            cfg['NUM_DMA_READ_CLIENTS'] == 7

        if cfg.get('NUM_DMA_WRITE_CLIENTS', None) != None:
            assert cfg['NUM_DMA_WRITE_CLIENTS'] == 5 or \
            cfg['NUM_DMA_WRITE_CLIENTS'] == 3
        
        cfg['PDP_SINGLE_LBUF_WIDTH'] = int(16 * cfg['MEMORY_ATOMIC_SIZE'] / cfg['PDP_THROUGHPUT'])
        cfg['PDP_SINGLE_LBUF_DEPTH'] = int(cfg['PDP_THROUGHPUT'] * (cfg['BPE'] + 6))


        cfg['MAC_ATOMIC_C_SIZE_LOG2'] = int(math.log2(cfg['MAC_ATOMIC_C_SIZE']))
        cfg['MAC_ATOMIC_K_SIZE_LOG2'] = int(math.log2(cfg['MAC_ATOMIC_K_SIZE']))
        cfg['ATOMIC_K_SIZE_DIV2'] = int(cfg['MAC_ATOMIC_K_SIZE'] / 2)
        cfg['CBUF_BANK_NUMBER_LOG2'] = int(math.log2(cfg['CBUF_BANK_NUMBER']))
        cfg['CBUF_BANK_WIDTH_LOG2'] = int(math.log2(cfg['CBUF_BANK_WIDTH']))
        cfg['BANK_DEPTH_LOG2'] = int(math.log2(cfg['CBUF_BANK_DEPTH']))
        cfg['CBUF_DEPTH_LOG2 '] = int(math.log2(cfg['CBUF_BANK_NUMBER'])) + int(math.log2(cfg['CBUF_BANK_DEPTH']))
        cfg['CBUF_ENTRY_WIDTH'] = cfg['MAC_ATOMIC_C_SIZE'] * cfg['BPE']
        cfg['CBUF_WIDTH_LOG2'] = int(math.log2(cfg['CBUF_ENTRY_WIDTH']))
        cfg['CBUF_WIDTH_MUL2_LOG2'] = int(math.log2(cfg['CBUF_ENTRY_WIDTH'])) + 1
        cfg['BPE_LOG2'] = int(math.log2(cfg['BPE']))
        cfg['MAC_RESULT_WIDTH'] = cfg['BPE'] * 2 + cfg['MAC_ATOMIC_C_SIZE_LOG2']
        cfg['CC_ATOMC_DIV_ATOMK'] = int(cfg['MAC_ATOMIC_C_SIZE'] / cfg['MAC_ATOMIC_K_SIZE'])
        cfg['CACC_SDP_WIDTH'] = ((32 * cfg['SDP_MAX_THROUGHPUT']) +2)
        cfg['CACC_SDP_SINGLE_THROUGHPUT'] = 32
        
        cfg['CDMA_GRAIN_MAX_BIT'] = int(math.log2(cfg['CBUF_BANK_DEPTH'] * cfg['CBUF_BANK_WIDTH'] * \
            (cfg['CBUF_BANK_NUMBER']-1) / (cfg['MEMORY_ATOMIC_SIZE'])))


        # Macro defs
        self.macro_defs = []
        self.cfg_dict = {}
        self.headers_list = []

        for key in cfg:
            self.macro_defs.append("-DNVDLA_%s=%s" % (key, str(cfg[key])))
            self.headers_list.append("#define NVDLA_%s %s\n" % (key, str(cfg[key])))
            self.cfg_dict[key] = cfg[key]

    @property
    def cflags(self):
        return self.macro_defs

    @property
    def TARGET(self):
        return self.cfg_dict['target']

    @property
    def headers(self):
        return self.headers_list

    @property
    def cfg_json(self):
        return json.dumps(self.cfg_dict, indent=2)

    def same_config(self, cfg):
        """Compare if cfg is same as current config.

        Parameters
        ----------
        cfg : the configuration
            The configuration

        Returns
        -------
        equal : bool
            Whether the configuration is the same.
        """
        for k, v in self.cfg_dict.items():
            if k not in cfg:
                return False
            if cfg[k] != v:
                return False
        return True
