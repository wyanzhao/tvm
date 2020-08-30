#include "nvdla_api.h"
#include "half.h"
#include "ErrorMacros.h"
#include "priv/caffe/CaffeParser.h"

#include <string>
#include <memory>
#include <cstring>
#include <set>
#include <unordered_map>

#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_DATA_FMT nvdla::DataFormat::NCHW
#define DEFAULT_QUANT_MODE nvdla::QuantizationMode::NONE
#define TARGET_CONFIG_NAME "nv_full"

// TODO: Remember to change configuratio here
static TestAppArgs defaultTestAppArgs =
{
    /* .project = */ "OpenDLA",
    /* .inputPath = */ "./",
    /* .inputName = */ "",
    /* .outputPath = */ "./",
    /* .testname = */ "",
    /* .testArgs = */ "",
    /* .prototxt = */ "",
    /* .caffemodel = */ "",
    /* .cachemodel = */ "",
    /* .profileName = */ "fast-math",
    /* .profileFile = */ "",
    /* .configtarget = */ TARGET_CONFIG_NAME,
    /* .calibtable = */ "",
    /* .quantizationMode = */ DEFAULT_QUANT_MODE,
    /* .numBatches = */ DEFAULT_BATCH_SIZE,
    /* .inDataFormat = */ DEFAULT_DATA_FMT,
    /* .computePrecision = */ nvdla::DataType::HALF
};

class ScaleInfo {
private:
    float m_min;
    float m_max;
    float m_scale;
public:
    float getMin() {return m_min;};
    float getMax() {return m_max;};
    float getScale() {return m_scale;};
    void setMin(float min){m_min = min;};
    void setMax(float max) {m_max = max;};
    void setScale(float scale) {m_scale = scale;};
};

static std::unordered_map<std::string, ScaleInfo*> scale_map;


#ifdef __cplusplus
extern "C" {
#endif


static TestAppArgs testAppArgs = defaultTestAppArgs;
static TestInfo* testInfo = nullptr;
static nvdla::INetwork* network = nullptr;
static nvdla::caffe::IBlobNameToTensor* blobNameToTensor = nullptr;

void setNvdlaConfig( const char* config_name, const char *cprecision)
{
    std::string computePrecision(cprecision);

    if (computePrecision == "fp16")
        testAppArgs.computePrecision = nvdla::DataType::HALF;
    else if (computePrecision == "int8")
        testAppArgs.computePrecision = nvdla::DataType::INT8;
    else {
        testAppArgs.computePrecision = nvdla::DataType::UNKNOWN;
        std::cout<< "UNKNOWN Compute Precision"<< std::endl;
        }
        
    testAppArgs.configtarget = std::string(config_name);

    std::cout<< "Current configuration: "<< testAppArgs.configtarget<< " "<< computePrecision<< std::endl;
}

void addScaleInfo(const char * name, float scale, float min, float max, int offset)
{   
    std::string name_(name);
    ScaleInfo* scale_info = new(ScaleInfo);
    scale_info->setMin(min);
    scale_info->setMax(max);
    scale_info->setScale(scale);
    auto it = scale_map.find(name_);
    if (it == scale_map.end())
    {
        std::pair<std::string, ScaleInfo*> name_pair (name_, scale_info);
        scale_map.insert(name_pair);
    }
}

NvDlaError addQuantizationInfo(const TestAppArgs* appArgs, TestInfo *i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;

    // populate the scaling factor/dynamic range of each of the tensors on the network
    {
        {
            std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
            std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

            std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
            std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

            // set scaling factor for the network input tensors
            for (; nii != networkInputs.end(); ++nii)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string tName = (*nii)->getName();
                auto it = scale_map.find(tName);
                if (it != scale_map.end())
                {
                    ScaleInfo* scale_info = it->second;
                    scale = scale_info->getScale();
                    if(scale != 0)
                    {
                        min = scale * -127.0f;
                        max = scale * 127.0f;
                    } else {
                        min = scale_info->getMin() * -127.0f;
                        max = scale_info->getMax() * 127.0f;
                    }
                    
                } else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", tName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                (*nii)->setChannelDynamicRange(-1, min, max);
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
            }

            for (; li != networkLayers.end(); ++li)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string lName = (*li)->getName();
                nvdla::ITensor* outTensor = (*li)->getOutput(0);
                auto it = scale_map.find(lName);
                if (it != scale_map.end())
                {
                    ScaleInfo* scale_info = it->second;
                    scale = scale_info->getScale();
                    if(scale != 0)
                    {
                        min = scale * -127.0f;
                        max = scale * 127.0f;
                    } else {
                        min = scale_info->getMin() * -127.0f;
                        max = scale_info->getMax() * 127.0f;
                    }
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", lName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                outTensor->setChannelDynamicRange(-1, min, max);
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
            }
        }
    }

fail:
    return e;
}


NvDlaError parseTensorScales(const TestAppArgs* appArgs, TestInfo *i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string calibTableFile = /*i->calibTablesPath + "/" + */appArgs->calibTable;

    PROPAGATE_ERROR_FAIL(NvDlaStat(calibTableFile.c_str(), &stat));

    // populate the scaling factor/dynamic range of each of the tensors on the network
    {
        FILE* fp = fopen(calibTableFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
            std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

            std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
            std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

            // set scaling factor for the network input tensors
            for (; nii != networkInputs.end(); ++nii)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string tName = (*nii)->getName();
                if (doc[tName.c_str()].HasMember("scale")) {
                    scale = doc[tName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[tName.c_str()].HasMember("min") && doc[tName.c_str()].HasMember("max")) {
                    min = doc[tName.c_str()]["min"].GetFloat();
                    max = doc[tName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", tName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
            }

            for (; li != networkLayers.end(); ++li)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string lName = (*li)->getName();
                nvdla::ITensor* outTensor = (*li)->getOutput(0);

                if (doc[lName.c_str()].HasMember("scale")) {
                    scale = doc[lName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[lName.c_str()].HasMember("min") && doc[lName.c_str()].HasMember("max")) {
                    min = doc[lName.c_str()]["min"].GetFloat();
                    max = doc[lName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", lName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
            }
        }

        fclose(fp);
    }

fail:
    return e;
}


static NvDlaError beginWithNamedProfile(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profiler not initialized");
    }

    profile = profiler->getProfile(appArgs->profileName.c_str());
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profile %s not initialized", appArgs->profileName.c_str());
    }

fail:
    return e;
}

static NvDlaError beginWithCfgProfile(const TestAppArgs* appArgs, TestInfo* i, nvdla::DataFormat& inDataFormat)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string profileCfgFile;
    std::string profileName;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profiler not initialized");
    }

    profileName = appArgs->profileFile;
    profileName = profileName.substr(0, profileName.find_last_of("."));
    profile = profiler->getProfile(profileName.c_str());
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "Profile %s not initialized", profileName.c_str());
    }

    profileCfgFile = i->profilesPath + appArgs->profileFile;
    PROPAGATE_ERROR_FAIL(NvDlaStat(profileCfgFile.c_str(), &stat));

    // first use settings from default profile
    profile->initWithDefaultProfile();

    // then populate the existing profile with params in the cfg file (overriding as necessary)
    {
        FILE* fp = fopen(profileCfgFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            nvdla::PixelFormat pf;

            /* Gather compile params of the profile */
            if (doc["profile"].HasMember("compute_precision")) {
                rapidjson::Value& compPrecision = doc["profile"]["compute_precision"];
                std::string prec = compPrecision.GetString();
                nvdla::DataType dt = nvdla::DataType::getEnum(prec);

                if (dt.v() == nvdla::DataType::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Precision %s not supported", prec.c_str());
                }
                profile->setComputePrecision(dt);
            }

            if (doc["profile"].HasMember("weight_packing")) {
                rapidjson::Value& weightPacking = doc["profile"]["weight_packing"];
                std::string wtPacking = weightPacking.GetString();

                if ( wtPacking == "COMPRESSED" )
                    profile->setCanCompressWeights(true);
                else
                    profile->setCanCompressWeights(false);
            }

            if (doc["profile"].HasMember("sdp_pdp_on_fly")) {
                rapidjson::Value& sdpPdpOnFly = doc["profile"]["sdp_pdp_on_fly"];
                profile->setCanSDPPDPOnFly(sdpPdpOnFly.GetBool());
            }

            if (doc["profile"].HasMember("sdp_merge_math_ops")) {
                rapidjson::Value& sdpMergeMathOps = doc["profile"]["sdp_merge_math_ops"];
                profile->setCanSDPMergeMathOps(sdpMergeMathOps.GetBool());
            }

            if (doc["profile"].HasMember("sdp_fuse_subengine_ops")) {
                rapidjson::Value& sdpFuseSubEngineOps = doc["profile"]["sdp_fuse_subengine_ops"];
                profile->setCanSDPFuseSubEngineOps(sdpFuseSubEngineOps.GetBool());
            }

            if (doc["profile"].HasMember("can_winograd")) {
                rapidjson::Value& canWinograd = doc["profile"]["can_winograd"];
                profile->setCanWinograd(canWinograd.GetBool());
            }

            /* Gather global params of the profile */
            if (doc["profile"]["network_input"].HasMember("format")) {
                rapidjson::Value& inFormat = doc["profile"]["network_input"]["format"];
                pf = nvdla::PixelFormat::getEnum(inFormat.GetString());
                if (pf.v() == nvdla::PixelFormat::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Pixel format %s not supported", inFormat.GetString());
                }
                profile->setNetworkInputSurfaceFormat(pf);
                if (pf < nvdla::PixelFormat::FEATURE) {
                    inDataFormat = nvdla::DataFormat::NHWC;
                }
                else if ((pf == nvdla::PixelFormat::FEATURE) || (pf == nvdla::PixelFormat::FEATURE_X8)) {
                    inDataFormat = nvdla::DataFormat::NCxHWx;
                }
                else {
                    PROPAGATE_ERROR_FAIL(NvDlaError_NotSupported, "Don't support input pixel format: %s", pf.c_str());
                }
            }

            if (doc["profile"]["network_input"].HasMember("pixel_offset_x")) {
                rapidjson::Value& pxOffX = doc["profile"]["network_input"]["pixel_offset_x"];
                profile->setNetworkInputPixelOffX(pxOffX.GetInt());
            }
            if (doc["profile"]["network_input"].HasMember("pixel_offset_y")) {
                rapidjson::Value& pxOffY = doc["profile"]["network_input"]["pixel_offset_y"];
                profile->setNetworkInputPixelOffY(pxOffY.GetInt());
            }

            if (doc["profile"]["network_output"].HasMember("format")) {
                rapidjson::Value& outFormat = doc["profile"]["network_output"]["format"];
                pf = nvdla::PixelFormat::getEnum(outFormat.GetString());
                if (pf.v() == nvdla::PixelFormat::UNKNOWN) {
                    ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Pixel format %s not supported", outFormat.GetString());
                }
                profile->setNetworkOutputSurfaceFormat(pf);
            }
        }

        fclose(fp);
    }

fail:
    return e;
}

static NvDlaError updateProfileWithCmdLineArgs
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    const char* profileName,
    nvdla::DataFormat inDataFormat
)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IProfiler* profiler;
    nvdla::IProfile* profile;

    profiler = i->wisdom->getProfiler();
    if (!profiler)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->getProfiler() failed");
    profile   = profiler->getProfile(profileName);
    if (!profile)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "profiler->getProfile() failed");

    PROPAGATE_ERROR_FAIL(profile->setComputePrecision(appArgs->computePrecision));
    PROPAGATE_ERROR_FAIL(profile->setNetworkInputDataFormat(inDataFormat));

    // determine input surface format
    switch(inDataFormat)
    {
        case nvdla::DataFormat::NHWC:

            if (appArgs->computePrecision == nvdla::DataType::HALF)
            {
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A16B16G16R16_F));
            }
            else if (appArgs->computePrecision == nvdla::DataType::INT8)
            {
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A8B8G8R8));
            }
            else
            {
                ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "NHWC and compute precision %u is not yet supported",
                                     appArgs->computePrecision.v());
            }
            break;
        case nvdla::DataFormat::NCxHWx:
        case nvdla::DataFormat::NCHW:
        case nvdla::DataFormat::UNKNOWN:    // atleast start the test with feature data format
        default:
            if (std::strcmp(appArgs->configtarget.c_str(), "opendla-small") == 0)
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8));
            else
                PROPAGATE_ERROR_FAIL(profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE));
    }

    // determine int8 cfgs
    if (appArgs->computePrecision == nvdla::DataType::INT8)
    {
        PROPAGATE_ERROR_FAIL(profile->setTensorScalingMode(nvdla::TensorScalingMode::PER_TENSOR));
        switch(appArgs->quantizationMode)
        {
            case nvdla::QuantizationMode::PER_FILTER:
                PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::PER_FILTER));
                break;
            case nvdla::QuantizationMode::PER_KERNEL:
            case nvdla::QuantizationMode::NONE: // default to per-kernel; find a way to run int8 tests w/ NONE qtzMode cleanly
            default:
                PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::PER_KERNEL));
        }
    }
    else
    {
        PROPAGATE_ERROR_FAIL(profile->setTensorScalingMode(nvdla::TensorScalingMode::NONE));
        PROPAGATE_ERROR_FAIL(profile->setQuantizationMode(nvdla::QuantizationMode::NONE));
    }

    PROPAGATE_ERROR_FAIL(profile->setNetworkOutputDataFormat(nvdla::DataFormat::NCxHWx));

    if (std::strcmp(appArgs->configtarget.c_str(), "opendla-small") == 0)
        PROPAGATE_ERROR_FAIL(profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8));
    else
        PROPAGATE_ERROR_FAIL(profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE));

    if (appArgs->numBatches > 0)
        PROPAGATE_ERROR_FAIL(profile->setMultiBatchSize(appArgs->numBatches));

fail:
    return e;
}

NvDlaError generateProfile(const TestAppArgs* appArgs, std::string* profileName, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::DataFormat inDataFormat = nvdla::DataFormat::UNKNOWN;

    if (appArgs->profileName != "")
    {
        // init named profile (basic/default/performance) with default params in its constructor and exit
        PROPAGATE_ERROR_FAIL(beginWithNamedProfile(appArgs, i));
        *profileName = appArgs->profileName;
    }
    else if (appArgs->profileFile != "")
    {
        // if named profile is absent, create a default profile
        // and then populate it with params from the cfg file (overriding as necessary)
        PROPAGATE_ERROR_FAIL(beginWithCfgProfile(appArgs, i, inDataFormat));
        *profileName = appArgs->profileFile;
        *profileName = profileName->substr(0, profileName->find_last_of("."));
    }
    else
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No profile supplied to load");
    }

    // capture profile params from command line (override the existing ones as necessary)
    inDataFormat = inDataFormat == nvdla::DataFormat::UNKNOWN ? appArgs->inDataFormat : inDataFormat;
    PROPAGATE_ERROR_FAIL(updateProfileWithCmdLineArgs(appArgs, i, profileName->c_str(), inDataFormat));

fail:
    return e;
}

NvDlaError generateTensorScales(const TestAppArgs* appArgs, TestInfo* i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;

    std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
    std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

    std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
    std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

    // set scaling factor for the network input tensors
    for (; nii != networkInputs.end(); ++nii)
    {
        NvF32 scale = 1;
        NvF32 min = scale * -127.0f;
        NvF32 max = scale * 127.0f;
        std::string tName = (*nii)->getName();

        // set same dynamic range for all channels of the tensor (cIndex = -1)
        PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
        const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
        if (0)
            NvDlaDebugPrintf("setting dynamic range of: %s to %f\n", tName.c_str(), scale);
    }

    for (; li != networkLayers.end(); ++li)
    {
        NvF32 scale = 127;
        NvF32 min = scale * -127.0f;
        NvF32 max = scale * 127.0f;
        std::string lName = (*li)->getName();
        nvdla::ITensor* outTensor = (*li)->getOutput(0);

        // set same dynamic range for all channels of the tensor (cIndex = -1)
        PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
        const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
        if (0)
            NvDlaDebugPrintf("setting dynamic range of: %s to %f\n", lName.c_str(), scale);
    }

fail:
    return e;
}


static NvDlaError compileProfile(const TestAppArgs* appArgs, TestInfo* i)
{
        NvDlaError e = NvDlaSuccess;
    std::string profileName = "";
    std::string targetConfigName = "";

    NvDlaFileHandle file = 0;
    std::string fileName = "";
    NvU8 *buffer = 0;
    NvU64 size = 0;

    nvdla::ICompiler* compiler = i->wisdom->getCompiler();
    if (!compiler)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->getCompiler() failed");

    if (!(appArgs->configtarget != ""))
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No target config found to load");

    targetConfigName = appArgs->configtarget;

    // Determine profile
    PROPAGATE_ERROR_FAIL(generateProfile(appArgs, &profileName, i));

    // Compile
    NvDlaDebugPrintf("compiling profile \"%s\"... config \"%s\"...\n", profileName.c_str(), targetConfigName.c_str());
    PROPAGATE_ERROR_FAIL(compiler->compile(profileName.c_str(), targetConfigName.c_str(), &i->compiledLoadable));

    // Get loadable buffer and dump it into a file
    PROPAGATE_ERROR_FAIL(compiler->getLoadableImageSize(profileName.c_str(),
                                                    &size));
    if (size == 0) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter,
                            "Invalid size for a loadable");
    }

    buffer = (NvU8 *) NvDlaAlloc(size);
    if (buffer == NULL) {
        ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory,
                            "Failed to allocate buffer for loadable");
    }
    PROPAGATE_ERROR_FAIL(compiler->getLoadableImage(profileName.c_str(),
                                                    buffer));
    fileName = profileName + ".nvdla";
    PROPAGATE_ERROR_FAIL(NvDlaFopen(fileName.c_str(), NVDLA_OPEN_WRITE, &file));
    PROPAGATE_ERROR_FAIL(NvDlaFwrite(file, buffer, size));

fail:
    NvDlaFclose(file);
    if (buffer != NULL)
        NvDlaFree(buffer);
    return e;
}

void nvdlaInit()
{
    using namespace nvdla;
    NvDlaError e = NvDlaSuccess;
    testInfo = new(TestInfo);
    int ii = 0;

    std::string wisdomPath = testAppArgs.outputPath + "wisdom.dir/";
    std::string removeCmd = "";
    std::string imagePath = "";

    // Clear wisdomPath if any exist
    removeCmd += "rm -rf " + wisdomPath;
    ii = std::system(removeCmd.c_str()); // This is pretty awful
    if (ii != 0)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "system command failed: \"%s\"", removeCmd.c_str());

    PROPAGATE_ERROR_FAIL(NvDlaMkdir(const_cast<char *>(wisdomPath.c_str())));


    // Initialize TestInfo
    testInfo->wisdom = NULL;
    testInfo->wisdomPath = wisdomPath;
    testInfo->pData = NULL;

    
    NvDlaDebugPrintf("creating new wisdom context...\n");
    testInfo->wisdom = nvdla::createWisdom();
    if (!testInfo->wisdom)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createWisdom() failed");

    NvDlaDebugPrintf("opening wisdom context...\n");
    if (!testInfo->wisdom->open(testInfo->wisdomPath))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->open() failed to open: \"%s\"", testInfo->wisdomPath.c_str());


    network = nvdla::createNetwork();
    if (!network)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createNetwork() failed");

    blobNameToTensor = new nvdla::caffe::priv::BlobNameToTensor();
    if (!blobNameToTensor)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "newBlobNameToTensor() failed");

fail:
    return;
}

static int identifyOutputs(nvdla::INetwork * network)
{
    using namespace nvdla;

    std::set< ITensor* > outputTensors;
    std::set< ITensor* > inputTensors;

    for (int l = 0; l < network->getNumLayers(); ++l)
    {
        ILayer* layer = network->getLayer(l);
        if (!layer)
            return -1;

        for (int ii = 0; ii < layer->getNumInputs(); ++ii) {
            inputTensors.insert(layer->getInput(ii));
        }

        for (int oo = 0; oo < layer->getNumOutputs(); ++oo)
        {
            outputTensors.insert(layer->getOutput(oo));
        }
    }

    for (std::set<ITensor*>::iterator oi = outputTensors.begin(); oi != outputTensors.end(); ++oi)
    {
        // an output tensor which is not an input to any other layers is a network output tensor
        if (inputTensors.find(*oi) == inputTensors.end())
        {
            network->markOutput(*oi);
            gLogInfo << "mark " << (*oi)->getName() << std::endl;
        }
    }

    return network->getNumOutputs();
}

NvDlaError nvdlaCompile()
{
    NvDlaError e = NvDlaSuccess;
    
    // if the application has so far not marked the network's outputs, allow the parser to do so now
    if (network->getNumOutputs() <= 0)
    {
        int outs = identifyOutputs(network);
        NvDlaDebugPrintf("Marking total %d outputs\n", outs);
        if (outs <= 0)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Unable to identify outputs for the network: %d", outs);
    }

    if (testAppArgs.computePrecision == nvdla::DataType::INT8)
    {
        PROPAGATE_ERROR_FAIL(addQuantizationInfo(&testAppArgs, testInfo, network));
        // if (testAppArgs.calibTable != "")  // parse and set tensor scales
        // {
        //     NvDlaDebugPrintf("parsing calibration table...\n");
        //     PROPAGATE_ERROR_FAIL(parseTensorScales(&testAppArgs, testInfo, network));
        // }
        // else    // use default or const scaling factors
        // {
            // NvDlaDebugPrintf("initialize all tensors with const scaling factors of 127...\n");
             //PROPAGATE_ERROR_FAIL(generateTensorScales(&testAppArgs, testInfo, network));
        // }
    }

    NvDlaDebugPrintf("attaching parsed network to the wisdom...\n");
    if (!testInfo->wisdom->setNetworkTransient(network))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "wisdom->setNetworkTransient() failed");

     // Compile
    PROPAGATE_ERROR_FAIL(compileProfile(&testAppArgs, testInfo));

    /* Destroy network before closing wisdom context */
    nvdla::destroyNetwork(testInfo->wisdom->getNetwork());

    NvDlaDebugPrintf("closing wisdom context...\n");
    testInfo->wisdom->close();

fail:
    if (testInfo->wisdom != NULL) {
        nvdla::destroyWisdom(testInfo->wisdom);
        testInfo->wisdom = NULL;
    }

    return e;
}


void addInputOp(const char*  input_name, int n, int c, int h, int w)
{   
    nvdla::Dims4 dims(n, c, h, w);
    std::string input_name_((input_name));

    auto input_tensor = network->addInput(input_name_.c_str(), dims);
    blobNameToTensor->add(input_name_.c_str(), input_tensor);
}

nvdla::Weights* addFloatWeights(const void* values, uint64_t count)
{
    if (count == 0)
        return new nvdla::Weights(nvdla::DataType::FLOAT, nullptr, 0);

    float* weight_values = new float[count];

    std::memcpy(weight_values, values, sizeof(float) * count);

    nvdla::Weights* weights = new nvdla::Weights(nvdla::DataType::FLOAT, weight_values, count);

    return weights;
}


void addConvOp(const char* input_name, const char* op_name, int numOutputChannels,
                                           int kernelH, int kernelW, 
                                           int padH, int padW, 
                                           int strideH, int strideW,
                                           int dilationH, int dilationW,
                                           const nvdla::Weights* weights, const nvdla::Weights* bias_weights, int numGroups)
{
    std::string input_name_((input_name));

    std::string op_name_((op_name));

    assert(weights != nullptr && bias_weights != nullptr);

    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;
    
    if ( bias_weights->count == 0 )
    {
        biasMode = nvdla::BiasMode::bNONE;
    }
    else if ( bias_weights->count == 1 )
    {
        biasMode = nvdla::BiasMode::bUNIFORM;
    }
    else if ( bias_weights->count == numOutputChannels )
    {
        biasMode = nvdla::BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

    nvdla::Dims2 tlPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 brPadding = nvdla::Dims2(padH, padW);
    nvdla::Dims2 stride    = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 dilation  = nvdla::Dims2(dilationH, dilationW);
    nvdla::Dims2 kernelSize= nvdla::Dims2(kernelH, kernelW);

    auto input_tensor = blobNameToTensor->find(input_name_.c_str());

    // TODO: cross-correlation vs convolution
    auto conv_op = network->addConvolution(input_tensor, numOutputChannels, 0,
                                    kernelSize, tlPadding, brPadding, stride, dilation,
                                    *weights, *bias_weights, biasMode, numGroups);

    conv_op->setName(op_name_.c_str());
    blobNameToTensor->add(op_name_.c_str(), conv_op->getOutput(0));
}


void addFullyConnected(const char*  input_name, const char*  op_name, const nvdla::Weights* weights,
const nvdla::Weights* bias_weights, int64_t num_output)
{
    std::string input_name_((input_name));
    std::string op_name_((op_name));

    assert(weights != nullptr && bias_weights != nullptr);

    nvdla::BiasMode biasMode = nvdla::BiasMode::bNONE;
    if ( bias_weights->count == 0 )
    {
        biasMode = nvdla::BiasMode::bNONE;
    }
    else if ( bias_weights->count == 1 )
    {
        biasMode = nvdla::BiasMode::bUNIFORM;
    }
    else if ( bias_weights->count == num_output )
    {
        biasMode = nvdla::BiasMode::bCHANNEL;
    }
    else
    {
        biasMode = nvdla::BiasMode::bm_ELEMENTWISE;
    }

    auto input_tensor = blobNameToTensor->find(input_name_.c_str());
    auto fc_op = network->addFullyConnected(input_tensor,
    num_output, *weights, *bias_weights, biasMode);

    fc_op->setName(op_name_.c_str());
    blobNameToTensor->add(op_name_.c_str(), fc_op->getOutput(0));
}


void addReluOp(const char* input_name, const char* op_name)
{
    std::string input_name_(input_name);

    auto input_tensor = blobNameToTensor->find(input_name_.c_str());

    std::string op_name_(op_name);

    auto relu_op = network->addActivation(input_tensor, /*ActivationType::*/nvdla::kRELU);
    relu_op->setName(op_name_.c_str());
    blobNameToTensor->add(op_name_.c_str(), relu_op->getOutput(0));
}


void addSoftMaxOp(const char*  input_name, const char*  op_name)
{
    std::string input_name_((input_name));

    auto input_tensor = blobNameToTensor->find(input_name_.c_str());

    std::string op_name_((op_name));

    auto op_ = network->addSoftMax(input_tensor);
    op_->setName(op_name_.c_str());
    blobNameToTensor->add(op_name_.c_str(), op_->getOutput(0));
}



 void addMaxPooling(const char*  input_name, const char*  op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, int has_global_pooling)
{
    if (has_global_pooling)
        return addPooling(input_name, op_name, kernelH, kernelW, padH, padW, strideH, strideW, nvdla::PoolingType::kMAX, true);
    else
        return addPooling(input_name, op_name, kernelH, kernelW, padH, padW, strideH, strideW, nvdla::PoolingType::kMAX, false);
}

 void addAveragePooling(const char*  input_name, const char*  op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, int has_global_pooling)
{
    if (has_global_pooling)
        return addPooling(input_name, op_name, kernelH, kernelW, padH, padW, strideH, strideW, nvdla::PoolingType::kAVERAGE, true);
    else
        return addPooling(input_name, op_name, kernelH, kernelW, padH, padW, strideH, strideW, nvdla::PoolingType::kAVERAGE, false);
}


static void addPooling(const char* input_name, const char* op_name, int kernelH, int kernelW,
                        int padH, int padW, int strideH, int strideW, nvdla::PoolingType type, bool has_global_pooling)
{
    std::string input_name_((input_name));

    std::string op_name_((op_name));
    if (has_global_pooling){
        printf("Currently doesn't support global pooling\n");
        exit -1;
    }

    nvdla::Dims2 windowSize = nvdla::Dims2(kernelH, kernelW);
    nvdla::Dims2 stride     = nvdla::Dims2(strideH, strideW);
    nvdla::Dims2 tlPadding  = nvdla::Dims2(padH, padW);
    nvdla::Dims2 brPadding  = nvdla::Dims2(padH, padW);

    auto input_tensor = blobNameToTensor->find(input_name_.c_str());
    // TODO: cross-correlation vs convolution
    auto op = network->addPooling(input_tensor, type, windowSize, stride, tlPadding, brPadding);

    op->setName(op_name_.c_str());
    blobNameToTensor->add(op_name_.c_str(), op->getOutput(0));
}


#ifdef __cplusplus
}
#endif