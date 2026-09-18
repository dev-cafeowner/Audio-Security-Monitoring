#ifndef AUDIO_PLAYBACK_H
#define AUDIO_PLAYBACK_H

#include "xil_types.h"

int AudioPlayback_Init(void);
int AudioPlayback_StartDemo(const s16 *MonoSamples);
void AudioPlayback_Service(void);
int AudioPlayback_IsActive(void);
int AudioPlayback_HadError(void);
void AudioPlayback_WaitUntilDone(void);
void AudioPlayback_Stop(void);

#endif /* AUDIO_PLAYBACK_H */
