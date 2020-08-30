#ifndef _NVDLA_INTERFACE_H
#define _NVDLA_INTERFACE_H


#include "dlaerror.h"
#include "dlatypes.h"
// #include "priv/Type.h"
#include "nvdla/ILoadable.h"
#include "nvdla/IRuntime.h"
#include "nvdla/ITensor.h"
#include "nvdla/IWisdom.h"
#include "DlaImageUtils.h"
#include "nvdla_inf.h"
#include "nvdla/ICompiler.h"
#include "nvdla/ITargetConfig.h"
#include "half.h"
#include "nvdla/ILayer.h"
#include "nvdla/INetwork.h"
#include "nvdla/IProfile.h"
#include "nvdla/IProfiler.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"
// #include "ErrorMacros.h"
#include "nvdla_os_inf.h"

#include <string>


#define TEST_PARAM_FILE_MAX_SIZE    65536

enum TestImageTypes
{
    IMAGE_TYPE_PGM = 0,
    IMAGE_TYPE_JPG = 1,
    IMAGE_TYPE_UNKNOWN = 2,
};

struct TaskStatus
{
    NvU64 timestamp;
    NvU32 status_engine;
    NvU16 subframe;
    NvU16 status_task;
};

struct SubmitContext
{
    NvU32 id;
    std::string inputName;
    std::map < NvU16, std::vector<NvDlaImage*> > inputImages;
    std::vector<void *> inputBuffers;
    std::vector<void *> inputTaskStatus;
    std::map < NvU16, std::vector<NvDlaImage*> > outputImages;
    std::vector<NvDlaImage*> debugImages;
    std::vector<void *> debugBuffers;
    std::vector<void *> outputBuffers;
    std::vector<void *> outputTaskStatus;
};

struct TestAppArgs
{
    std::string project;
    std::string inputPath;
    std::string inputName;
    std::string outputPath;
    std::string testname;
    std::string testArgs;
    std::string prototxt; // This should be folded into testArgs
    std::string caffemodel; // This should be folded into testArgs
    std::string cachemodel; // This should be folded into testArgs

    std::string profileName; // ok here?
    std::string profileFile;
    std::string configtarget;
    std::string calibTable;
    nvdla::QuantizationMode quantizationMode;

    NvU16 numBatches;
    nvdla::DataFormat inDataFormat;
    nvdla::DataType computePrecision;

    std::map<std::string, NvF32> tensorScales;
};

struct TestInfo
{
    // common
    nvdla::IWisdom* wisdom;
    std::string wisdomPath;

    // parse
    std::string modelsPath;
    std::string profilesPath;
    std::string calibTablesPath;

    // runtime
    nvdla::IRuntime* runtime;
    nvdla::ILoadable* compiledLoadable;
    NvU8 *pData;
    std::string inputImagesPath;
    std::string inputLoadablePath;
    std::map<std::string, NvDlaImage*> inputImages;
    std::map<std::string, void *> inputBuffers;
    std::map<std::string, NvDlaImage*> outputImages;
    std::map<std::string, void *> outputBuffers;
    std::vector<SubmitContext*> submits;
    NvU32 timeout;
    NvU16 numBatches; // runtime's point-of-view
    NvU32 numSubmits;
};


#ifdef __cplusplus
extern "C" {
#endif


void nvdlaInit();
void addInputOp(const char*  input_name, int n, int c, int h, int w);
nvdla::Weights* addFloatWeights(const void* values, uint64_t count);
void addConvOp(const char*  input_name, const char*  op_name, int numOutputChannels,
                                           int kernelH, int kernelW, 
                                           int padH, int padW, 
                                           int strideH, int strideW,
                                           int dilationH, int dilationW,
                                           const nvdla::Weights* weights, const nvdla::Weights* bias_weights, int numGroups);

void addReluOp(const char*  input_name, const char*  op_name);
void addSoftMaxOp(const char*  input_name, const char*  op_name);
void addFullyConnected(const char*  input_name, const char*  op_name, const nvdla::Weights* weights,
                               const nvdla::Weights* bias_weights, int64_t num_output);

 void addMaxPooling(const char*  input_name, const char*  op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, int has_global_pooling);

void addAveragePooling(const char*  input_name, const char*  op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, int has_global_pooling);

static void addPooling(const char*  input_name, const char*  op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, nvdla::PoolingType type, bool has_global_pooling);

NvDlaError nvdlaCompile();
void nvdlaDoNothing() {;};
void addScaleInfo(const char * name, float scale, float min, float max, int offset);
NvDlaError addQuantizationInfo(const TestAppArgs* appArgs, TestInfo *i, nvdla::INetwork* network);
void setNvdlaConfig( const char* config_name, const char *cprecision);

#ifdef __cplusplus
}
#endif

#endif