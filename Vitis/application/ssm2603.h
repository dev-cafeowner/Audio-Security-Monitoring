#ifndef SSM2603_H
#define SSM2603_H

#include "xil_types.h"

int SSM2603_InitMic(void);
int SSM2603_InitPlayback(void);
int SSM2603_WriteReg(u8 RegAddr, u16 RegData);
int SSM2603_SetMicBoost(int Enable);

#endif /* SSM2603_H */
