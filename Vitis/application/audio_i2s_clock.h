#ifndef AUDIO_I2S_CLOCK_H
#define AUDIO_I2S_CLOCK_H

#include "xi2stx.h"
#include "xi2stx_hw.h"
#include "xstatus.h"

#define AUDIO_I2S_TX_CORE_VERSION 0x00010000U

/*
 * Initialize the PL I2S transmitter as the shared BCLK/LRCLK master.
 *
 * A Vitis IDE process can keep the previously generated BSP open while a new
 * XSA/bitstream is built.  In that transition state the BSP says IsMaster=0,
 * so XI2s_Tx_CfgInitialize() rejects the new master-mode hardware in its
 * self-test.  Validate the immutable hardware fields directly and construct
 * the public driver instance only for that one known metadata mismatch.
 * A platform regenerated from the current XSA continues to use the normal
 * vendor initialization path.
 */
static inline int AudioI2sClock_InitializeMaster(
    XI2s_Tx *InstancePtr, XI2stx_Config *ConfigPtr)
{
    u32 HardwareConfig;
    u32 HardwareChannels;
    u32 HardwareWidth;

    if ((InstancePtr == NULL) || (ConfigPtr == NULL)) {
        return XST_FAILURE;
    }
    if (ConfigPtr->IsMaster) {
        return XI2s_Tx_CfgInitialize(InstancePtr, ConfigPtr,
            ConfigPtr->BaseAddress);
    }

    InstancePtr->Config = *ConfigPtr;
    InstancePtr->Config.IsMaster = TRUE;
    InstancePtr->Config.BaseAddress = ConfigPtr->BaseAddress;
    InstancePtr->IsReady = 0U;
    InstancePtr->IsStarted = 0U;

    if (XI2s_Tx_GetVersion(InstancePtr) != AUDIO_I2S_TX_CORE_VERSION) {
        return XST_FAILURE;
    }
    HardwareConfig = XI2s_Tx_ReadReg(ConfigPtr->BaseAddress,
        XI2S_TX_CORE_CFG_OFFSET);
    HardwareChannels = (HardwareConfig & XI2S_TX_REG_CFG_NUM_CH_MASK) >>
        XI2S_TX_REG_CFG_NUM_CH_SHIFT;
    HardwareWidth = ((HardwareConfig & XI2S_TX_REG_CFG_DWDTH_MASK) != 0U) ?
        24U : 16U;
    if (((HardwareConfig & XI2S_TX_REG_CFG_MSTR_MASK) == 0U) ||
        (HardwareChannels != (2U * ConfigPtr->MaxNumChannels)) ||
        (HardwareWidth != ConfigPtr->DWidth)) {
        return XST_FAILURE;
    }

    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;
    XI2s_Tx_Enable(InstancePtr, FALSE);
    return XST_SUCCESS;
}

#endif
