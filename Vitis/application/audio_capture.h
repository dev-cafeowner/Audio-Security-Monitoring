#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include "xil_types.h"

int AudioCapture_Init(void);
int AudioCapture_RunToNpuInput(void);
void AudioCapture_PrintSummary(void);

int AudioCapture_StartContinuous(void);
int AudioCapture_ServiceContinuous(void);
int AudioCapture_HasWindow(void);
int AudioCapture_SnapshotWindowToNpuInput(void);
void AudioCapture_StopContinuous(void);
int AudioCapture_IsContinuousActive(void);
u32 AudioCapture_GetWindowSequence(void);
u64 AudioCapture_GetCapturedSamples(void);

#endif /* AUDIO_CAPTURE_H */
