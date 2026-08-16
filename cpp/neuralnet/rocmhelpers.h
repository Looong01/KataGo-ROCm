#ifndef NEURALNET_ROCMHELPERS_H_
#define NEURALNET_ROCMHELPERS_H_

#include "../neuralnet/rocmincludes.h"
#include "../neuralnet/activations.h"

// Stream type used by the shared kernel launchers (cudaandrocmhelpers.inc) and by the launch-stream
// setter declared in cudaandrocmhelpers.h. hipStream_t here; cudaStream_t on the CUDA side.
#define KATAGO_STREAM_T hipStream_t

#include "../neuralnet/cudaandrocmhelpers.h"

#endif  // NEURALNET_ROCMHELPERS_H_
