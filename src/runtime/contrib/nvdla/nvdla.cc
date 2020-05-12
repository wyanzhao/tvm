#include "nvdla.h"

namespace tvm {
namespace contrib {

using namespace runtime;


extern "C" void nvdlaVoid()
{
     nvdlaInit();
};



}};
