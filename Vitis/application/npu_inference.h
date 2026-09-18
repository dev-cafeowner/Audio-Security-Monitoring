#ifndef NPU_INFERENCE_H
#define NPU_INFERENCE_H

#include "xil_types.h"

#define NPU_INFERENCE_RESULT_VALUES 527U

typedef int (*NpuInferenceServiceHook)(void);

int NpuInference_Init(void);
int NpuInference_PrepareModel(void);
int NpuInference_Run(u32 *FinalOutputAddress);
void NpuInference_SetServiceHook(NpuInferenceServiceHook Hook);
u32 NpuInference_GetLastRunUs(void);
void NpuInference_PrintDemoResults(u32 FinalOutputAddress,
                                   const char *InputModeName);

#endif /* NPU_INFERENCE_H */
