#include "ssm2603.h"

#include "xparameters.h"
#include "xiic_l.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "sleep.h"

#define SSM2603_IIC_BASEADDR       XPAR_XIIC_0_BASEADDR
#define SSM2603_I2C_ADDR           0x1AU

#define REG_LEFT_HEADPHONE         0x02U
#define REG_RIGHT_HEADPHONE        0x03U
#define REG_ANALOG_PATH            0x04U
#define REG_DIGITAL_PATH           0x05U
#define REG_POWER                  0x06U
#define REG_DIGITAL_IF             0x07U
#define REG_SAMPLE_RATE            0x08U
#define REG_ACTIVE                 0x09U
#define REG_RESET                  0x0FU

#define POWER_MIC_EXTCLK           0x039U
#define POWER_DAC_EXTCLK_OUT_OFF   0x037U
#define POWER_DAC_EXTCLK           0x027U
#define ANALOG_MIC_NO_BOOST        0x004U
#define ANALOG_MIC_BOOST           0x005U
#define ANALOG_DAC_TO_OUTPUT       0x012U
#define DIGITAL_DAC_MUTED          0x008U
#define DIGITAL_DAC_UNMUTED        0x000U
#define DIGITAL_IF_I2S_SLAVE_16    0x002U
#define SAMPLE_RATE_32K            0x018U
#define HEADPHONE_0DB              0x079U

int SSM2603_WriteReg(u8 RegAddr, u16 RegData)
{
    u16 ControlWord = (((u16)RegAddr & 0x7FU) << 9) |
        (RegData & 0x01FFU);
    u8 TxBuffer[2];
    unsigned Sent;

    TxBuffer[0] = (u8)(ControlWord >> 8);
    TxBuffer[1] = (u8)ControlWord;
    Sent = XIic_Send(SSM2603_IIC_BASEADDR, SSM2603_I2C_ADDR,
        TxBuffer, sizeof(TxBuffer), XIIC_STOP);
    return (Sent == sizeof(TxBuffer)) ? XST_SUCCESS : XST_FAILURE;
}

static int WriteChecked(u8 RegAddr, u16 RegData, const char *Name)
{
    int Status = SSM2603_WriteReg(RegAddr, RegData);
    xil_printf("[SSM2603] %-10s %s R%02x=0x%03x\r\n", Name,
        (Status == XST_SUCCESS) ? "PASS" : "FAIL", RegAddr, RegData);
    return Status;
}

static int ResetCodec(void)
{
    XIic_WriteReg(SSM2603_IIC_BASEADDR, XIIC_RESETR_OFFSET,
        XIIC_RESET_MASK);
    usleep(1000U);
    if (WriteChecked(REG_RESET, 0x000U, "RESET") != XST_SUCCESS) {
        return XST_FAILURE;
    }
    usleep(1000U);
    return XST_SUCCESS;
}

static int ConfigureCommon(u16 InitialPower, u16 FinalPower,
                           u16 AnalogPath, u16 DigitalPath)
{
    int Status;

    if (WriteChecked(REG_POWER, InitialPower, "POWER") != XST_SUCCESS ||
        WriteChecked(REG_ANALOG_PATH, AnalogPath, "ANALOG") != XST_SUCCESS ||
        WriteChecked(REG_DIGITAL_PATH, DigitalPath, "DIGITAL") != XST_SUCCESS ||
        WriteChecked(REG_DIGITAL_IF, DIGITAL_IF_I2S_SLAVE_16,
            "I2S") != XST_SUCCESS ||
        WriteChecked(REG_SAMPLE_RATE, SAMPLE_RATE_32K,
            "32K RATE") != XST_SUCCESS) {
        return XST_FAILURE;
    }
    usleep(100000U);
    Status = WriteChecked(REG_ACTIVE, 0x001U, "ACTIVE");
    if ((Status == XST_SUCCESS) && (FinalPower != InitialPower)) {
        /* SSM2603 requires OUT to be enabled only after VMID and ACTIVE. */
        Status = WriteChecked(REG_POWER, FinalPower, "OUTPUT ON");
    }
    return Status;
}

int SSM2603_InitMic(void)
{
    xil_printf("[SSM2603] MIC mode: ADC on, +20 dB boost, DAC/output muted\r\n");
    if (ResetCodec() != XST_SUCCESS) {
        return XST_FAILURE;
    }
    return ConfigureCommon(POWER_MIC_EXTCLK, POWER_MIC_EXTCLK,
        ANALOG_MIC_BOOST, DIGITAL_DAC_MUTED);
}

int SSM2603_InitPlayback(void)
{
    xil_printf("[SSM2603] DEMO mode: DAC/headphone output on\r\n");
    if (ResetCodec() != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (WriteChecked(REG_LEFT_HEADPHONE, HEADPHONE_0DB,
            "HP LEFT") != XST_SUCCESS ||
        WriteChecked(REG_RIGHT_HEADPHONE, HEADPHONE_0DB,
            "HP RIGHT") != XST_SUCCESS) {
        return XST_FAILURE;
    }
    return ConfigureCommon(POWER_DAC_EXTCLK_OUT_OFF, POWER_DAC_EXTCLK,
        ANALOG_DAC_TO_OUTPUT, DIGITAL_DAC_UNMUTED);
}

int SSM2603_SetMicBoost(int Enable)
{
    return SSM2603_WriteReg(REG_ANALOG_PATH,
        Enable ? ANALOG_MIC_BOOST : ANALOG_MIC_NO_BOOST);
}
