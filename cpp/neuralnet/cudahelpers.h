#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

// Stream type used by the shared kernel launchers (cudaandrocmhelpers.inc) and by the launch-stream
// setter declared in cudaandrocmhelpers.h. cudaStream_t here; hipStream_t on the ROCm side.
#define KATAGO_STREAM_T cudaStream_t

#include "../neuralnet/cudaandrocmhelpers.h"

#endif  // NEURALNET_CUDAHELPERS_H_
