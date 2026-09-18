#include "audio_capture.h"

#include "audio_i2s_clock.h"
#include "audio_stream.h"
#include "npu_memory_map.h"
#include "xaudioformatter.h"
#include "xaudioformatter_hw.h"
#include "xi2srx.h"
#include "xi2stx.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtimer_config.h"
#include "sleep.h"

#include <string.h>

#define AF_BASEADDR                 XPAR_XAUDIO_FORMATTER_0_BASEADDR
#define I2S_RX_BASEADDR             XPAR_XI2SRX_0_BASEADDR
#define I2S_TX_BASEADDR             XPAR_XI2STX_0_BASEADDR
#define AUDIO_MCLK_HZ               12288000U
#define AUDIO_PERIOD_TIME_US        250000U
#define AUDIO_FIRST_PERIOD_GUARD_US 20000U
#define AF_S2MM_ERROR_MASK          (BIT(17) | BIT(18) | BIT(19) | BIT(30))

static XAudioFormatter AudioFormatter;
static XI2s_Rx I2sReceiver;
static XI2s_Tx ClockTransmitter;

static int ContinuousCaptureActive;
static int ContinuousCaptureError;
static u32 ContinuousLastRawTransferCount;
static u64 ContinuousTotalTransferBytes;
static u64 ContinuousNextPeriodToCopy;
static u64 ContinuousCapturedSamples;
static u64 ContinuousNextWindowEndSample;
static u32 ContinuousWindowSequence;
static XTime ContinuousLastServiceTime;

static s16 AudioCapture_SaturateS16(s64 Value, u32 *ClippedSamples)
{
    if (Value > 32767) {
        ++(*ClippedSamples);
        return 32767;
    }
    if (Value < -32768) {
        ++(*ClippedSamples);
        return -32768;
    }
    return (s16)Value;
}

static void AudioCapture_ConditionNpuInput(s16 *Samples)
{
    s64 Sum = 0;
    u64 RawSumAbs = 0U;
    u64 CenteredSumAbs = 0U;
    s32 DcOffset;
    u32 Peak = 0U;
    u32 ClippedSamples = 0U;
    u32 Index;

    for (Index = 0U; Index < AUDIO_MONO_SAMPLE_COUNT; ++Index) {
        s32 Value = Samples[Index];
        Sum += Value;
        RawSumAbs += (Value < 0) ? (u32)(-Value) : (u32)Value;
    }
    DcOffset = (s32)(Sum / (s64)AUDIO_MONO_SAMPLE_COUNT);

    for (Index = 0U; Index < AUDIO_MONO_SAMPLE_COUNT; ++Index) {
        s32 Centered = (s32)Samples[Index] - DcOffset;
        u32 Magnitude = (Centered < 0) ?
            (u32)(-Centered) : (u32)Centered;

        CenteredSumAbs += Magnitude;
        if (Magnitude > Peak) {
            Peak = Magnitude;
        }
    }

    for (Index = 0U; Index < AUDIO_MONO_SAMPLE_COUNT; ++Index) {
        s64 Centered = (s32)Samples[Index] - DcOffset;
        Samples[Index] = AudioCapture_SaturateS16(
            Centered, &ClippedSamples);
    }

    xil_printf("[CAPTURE] condition dc=%d raw_mean_abs=%u "
               "centered_mean_abs=%u peak=%u gain=%u.%03ux clipped=%u\r\n",
        (int)DcOffset,
        (unsigned)(RawSumAbs / AUDIO_MONO_SAMPLE_COUNT),
        (unsigned)(CenteredSumAbs / AUDIO_MONO_SAMPLE_COUNT),
        (unsigned)Peak,
        1U, 0U,
        (unsigned)ClippedSamples);
}

static s16 *AudioCapture_Ring(void)
{
    return (s16 *)(UINTPTR)AUDIO_CAPTURE_RING_BASE;
}

static s16 *AudioCapture_NpuInput(void)
{
    return (s16 *)(UINTPTR)NPU_INPUT_BASE;
}

static s16 *AudioCapture_History(void)
{
    return (s16 *)(UINTPTR)AUDIO_MIC_HISTORY_BASE;
}

static int AudioCapture_CheckStatus(void)
{
    u32 Status = XAudioFormatter_ReadReg(AF_BASEADDR,
        XAUD_FORMATTER_S2MM_OFFSET + XAUD_FORMATTER_STS);

    if ((Status & AF_S2MM_ERROR_MASK) != 0U) {
        xil_printf("[CAPTURE] S2MM error status=0x%08x\r\n",
            (unsigned)Status);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int AudioCapture_Init(void)
{
    XI2srx_Config *I2sConfig;
    XI2stx_Config *ClockConfig;
    int Status;

    Status = (int)XAudioFormatter_Initialize(&AudioFormatter, AF_BASEADDR);
    if ((Status != XST_SUCCESS) || !AudioFormatter.s2mm_presence) {
        xil_printf("[CAPTURE] Audio Formatter S2MM init failed\r\n");
        return XST_FAILURE;
    }
    AudioFormatter.ChannelId = XAudioFormatter_S2MM;

    I2sConfig = XI2s_Rx_LookupConfig((UINTPTR)I2S_RX_BASEADDR);
    if (I2sConfig == NULL) {
        xil_printf("[CAPTURE] I2S receiver config not found\r\n");
        return XST_FAILURE;
    }
    Status = XI2s_Rx_CfgInitialize(&I2sReceiver, I2sConfig,
        I2sConfig->BaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("[CAPTURE] I2S receiver init failed (%d)\r\n", Status);
        return XST_FAILURE;
    }
    XI2s_Rx_Enable(&I2sReceiver, FALSE);

    /* The verified design uses the TX IP as the common BCLK/LRCLK master.
     * In MIC mode its audio stream remains idle and the codec DAC is muted. */
    ClockConfig = XI2s_Tx_LookupConfig((UINTPTR)I2S_TX_BASEADDR);
    if (ClockConfig == NULL) {
        xil_printf("[CAPTURE] I2S clock transmitter config not found\r\n");
        return XST_FAILURE;
    }
    Status = AudioI2sClock_InitializeMaster(&ClockTransmitter, ClockConfig);
    if (Status != XST_SUCCESS) {
        xil_printf("[CAPTURE] I2S clock transmitter init failed (%d)\r\n",
            Status);
        return XST_FAILURE;
    }
    XI2s_Tx_Enable(&ClockTransmitter, FALSE);
    if (!ClockTransmitter.Config.IsMaster ||
        !ClockTransmitter.Config.Is32BitLR) {
        xil_printf("[CAPTURE] I2S TX must be 32-bit-slot master\r\n");
        return XST_FAILURE;
    }
    Status = (int)XI2s_Tx_SetSclkOutDiv(&ClockTransmitter,
        AUDIO_MCLK_HZ, AUDIO_SAMPLE_RATE_HZ);
    if (Status != XST_SUCCESS) {
        xil_printf("[CAPTURE] I2S clock divider failed (%d)\r\n", Status);
        return XST_FAILURE;
    }
    xil_printf("[CAPTURE] S2MM=0x%08x I2S-RX=0x%08x ring=0x%08x\r\n",
        (unsigned)AF_BASEADDR, (unsigned)I2S_RX_BASEADDR,
        (unsigned)AUDIO_CAPTURE_RING_BASE);
    return XST_SUCCESS;
}

static void AudioCapture_CopyContinuousPeriod(u64 AbsolutePeriod)
{
    s16 *Ring = AudioCapture_Ring();
    s16 *History = AudioCapture_History();
    u32 RingPeriod = (u32)(AbsolutePeriod %
        AUDIO_CAPTURE_STREAM_PERIOD_COUNT);
    u32 SourceBase = RingPeriod * AUDIO_CAPTURE_STREAM_PERIOD_FRAMES *
        AUDIO_CHANNELS;
    u32 Frame;

    Xil_DCacheInvalidateRange((INTPTR)&Ring[SourceBase],
        AUDIO_CAPTURE_STREAM_PERIOD_BYTES);
    for (Frame = 0U; Frame < AUDIO_CAPTURE_STREAM_PERIOD_FRAMES; ++Frame) {
        s32 Left = Ring[SourceBase + Frame * AUDIO_CHANNELS];
        s32 Right = Ring[SourceBase + Frame * AUDIO_CHANNELS + 1U];
        u32 HistoryIndex = (u32)(ContinuousCapturedSamples %
            AUDIO_MIC_HISTORY_SAMPLE_COUNT);

        History[HistoryIndex] = (s16)((Left + Right) / 2);
        ++ContinuousCapturedSamples;
    }
}

int AudioCapture_StartContinuous(void)
{
    XAudioFormatterHwParams Params;
    s16 *Ring = AudioCapture_Ring();
    s16 *History = AudioCapture_History();

    memset(Ring, 0, AUDIO_CAPTURE_STREAM_RING_BYTES);
    memset(History, 0, AUDIO_MIC_HISTORY_BYTES);
    Xil_DCacheFlushRange((INTPTR)Ring, AUDIO_CAPTURE_STREAM_RING_BYTES);

    ContinuousCaptureActive = 0;
    ContinuousCaptureError = 0;
    ContinuousLastRawTransferCount = 0U;
    ContinuousTotalTransferBytes = 0U;
    ContinuousNextPeriodToCopy = 0U;
    ContinuousCapturedSamples = 0U;
    ContinuousNextWindowEndSample = AUDIO_MONO_SAMPLE_COUNT;
    ContinuousWindowSequence = 0U;

    AudioFormatter.ChannelId = XAudioFormatter_S2MM;
    XAudioFormatterDMAStop(&AudioFormatter);
    XAudioFormatterDMAReset(&AudioFormatter);
    Params.buf_addr = (u64)AUDIO_CAPTURE_RING_BASE;
    Params.active_ch = AUDIO_CHANNELS;
    Params.bits_per_sample = BIT_DEPTH_16;
    Params.periods = AUDIO_CAPTURE_STREAM_PERIOD_COUNT;
    Params.bytes_per_period = AUDIO_CAPTURE_STREAM_PERIOD_BYTES;
    XAudioFormatterSetHwParams(&AudioFormatter, &Params);

    XAudioFormatterDMAStart(&AudioFormatter);
    XI2s_Rx_Enable(&I2sReceiver, TRUE);
    XI2s_Tx_Enable(&ClockTransmitter, TRUE);
    XTime_GetTime(&ContinuousLastServiceTime);
    ContinuousCaptureActive = 1;
    xil_printf("[CAPTURE] Continuous start: ring=%u bytes retention=%u ms "
               "window=10 s hop=%u s\r\n",
        (unsigned)AUDIO_CAPTURE_STREAM_RING_BYTES,
        (unsigned)AUDIO_CAPTURE_STREAM_RETENTION_MS,
        (unsigned)AUDIO_MIC_WINDOW_HOP_SEC);
    return XST_SUCCESS;
}

int AudioCapture_ServiceContinuous(void)
{
    XTime Now;
    u64 GapCounts;
    u64 GapMs;
    u64 CompletedPeriods;
    u32 RawTransferCount;
    u32 Delta;

    if (!ContinuousCaptureActive) {
        return ContinuousCaptureError ? XST_FAILURE : XST_SUCCESS;
    }
    if (AudioCapture_CheckStatus() != XST_SUCCESS) {
        ContinuousCaptureError = 1;
        AudioCapture_StopContinuous();
        return XST_FAILURE;
    }

    XTime_GetTime(&Now);
    GapCounts = (u64)(Now - ContinuousLastServiceTime);
    GapMs = (GapCounts * 1000U) / COUNTS_PER_SECOND;
    ContinuousLastServiceTime = Now;
    if ((ContinuousTotalTransferBytes != 0U) &&
        (GapMs >= AUDIO_CAPTURE_STREAM_RETENTION_MS)) {
        xil_printf("[CAPTURE] Continuous overrun: service gap=%u ms "
                   "retention=%u ms\r\n",
            (unsigned)GapMs,
            (unsigned)AUDIO_CAPTURE_STREAM_RETENTION_MS);
        ContinuousCaptureError = 1;
        AudioCapture_StopContinuous();
        return XST_FAILURE;
    }

    AudioFormatter.ChannelId = XAudioFormatter_S2MM;
    RawTransferCount = XAudioFormatterGetDMATransferCount(&AudioFormatter);
    if (RawTransferCount >= ContinuousLastRawTransferCount) {
        Delta = RawTransferCount - ContinuousLastRawTransferCount;
    } else {
        Delta = AUDIO_CAPTURE_STREAM_RING_BYTES -
            ContinuousLastRawTransferCount + RawTransferCount;
    }
    ContinuousLastRawTransferCount = RawTransferCount;
    ContinuousTotalTransferBytes += Delta;
    CompletedPeriods = ContinuousTotalTransferBytes /
        AUDIO_CAPTURE_STREAM_PERIOD_BYTES;

    if (CompletedPeriods > (ContinuousNextPeriodToCopy +
                            AUDIO_CAPTURE_STREAM_PERIOD_COUNT)) {
        xil_printf("[CAPTURE] Continuous overrun: completed=%u next=%u\r\n",
            (unsigned)CompletedPeriods,
            (unsigned)ContinuousNextPeriodToCopy);
        ContinuousCaptureError = 1;
        AudioCapture_StopContinuous();
        return XST_FAILURE;
    }
    while (ContinuousNextPeriodToCopy < CompletedPeriods) {
        AudioCapture_CopyContinuousPeriod(ContinuousNextPeriodToCopy);
        ++ContinuousNextPeriodToCopy;
    }
    return XST_SUCCESS;
}

int AudioCapture_HasWindow(void)
{
    return ContinuousCaptureActive &&
        (ContinuousCapturedSamples >= ContinuousNextWindowEndSample);
}

int AudioCapture_SnapshotWindowToNpuInput(void)
{
    s16 *History = AudioCapture_History();
    s16 *NpuInput = AudioCapture_NpuInput();
    u64 WindowStart;
    u32 Index;

    if (!AudioCapture_HasWindow()) {
        return XST_FAILURE;
    }
    if ((ContinuousCapturedSamples - ContinuousNextWindowEndSample) >
        (AUDIO_MIC_HISTORY_SAMPLE_COUNT - AUDIO_MONO_SAMPLE_COUNT)) {
        xil_printf("[CAPTURE] Snapshot expired: captured=%u end=%u\r\n",
            (unsigned)ContinuousCapturedSamples,
            (unsigned)ContinuousNextWindowEndSample);
        ContinuousCaptureError = 1;
        AudioCapture_StopContinuous();
        return XST_FAILURE;
    }

    WindowStart = ContinuousNextWindowEndSample - AUDIO_MONO_SAMPLE_COUNT;
    for (Index = 0U; Index < AUDIO_MONO_SAMPLE_COUNT; ++Index) {
        u32 HistoryIndex = (u32)((WindowStart + Index) %
            AUDIO_MIC_HISTORY_SAMPLE_COUNT);
        NpuInput[Index] = History[HistoryIndex];
    }
    AudioCapture_ConditionNpuInput(NpuInput);
    Xil_DCacheFlushRange((INTPTR)NpuInput, AUDIO_MONO_BYTES);
    ++ContinuousWindowSequence;
    xil_printf("[CAPTURE] Window %u: samples %u..%u -> NPU input 0x%08x\r\n",
        (unsigned)ContinuousWindowSequence,
        (unsigned)WindowStart,
        (unsigned)ContinuousNextWindowEndSample,
        (unsigned)NPU_INPUT_BASE);
    ContinuousNextWindowEndSample += AUDIO_MIC_WINDOW_HOP_SAMPLES;
    return XST_SUCCESS;
}

void AudioCapture_StopContinuous(void)
{
    AudioFormatter.ChannelId = XAudioFormatter_S2MM;
    XI2s_Rx_Enable(&I2sReceiver, FALSE);
    XI2s_Tx_Enable(&ClockTransmitter, FALSE);
    XAudioFormatterDMAStop(&AudioFormatter);
    ContinuousCaptureActive = 0;
}

int AudioCapture_IsContinuousActive(void)
{
    return ContinuousCaptureActive;
}

u32 AudioCapture_GetWindowSequence(void)
{
    return ContinuousWindowSequence;
}

u64 AudioCapture_GetCapturedSamples(void)
{
    return ContinuousCapturedSamples;
}

int AudioCapture_RunToNpuInput(void)
{
    XAudioFormatterHwParams Params;
    s16 *Ring = AudioCapture_Ring();
    s16 *NpuInput = AudioCapture_NpuInput();
    u32 Period;
    int Status = XST_SUCCESS;

    memset(Ring, 0, AUDIO_RING_BYTES);
    memset(NpuInput, 0, AUDIO_MONO_BYTES);
    Xil_DCacheFlushRange((INTPTR)Ring, AUDIO_RING_BYTES);

    AudioFormatter.ChannelId = XAudioFormatter_S2MM;
    XAudioFormatterDMAStop(&AudioFormatter);
    XAudioFormatterDMAReset(&AudioFormatter);
    Params.buf_addr = (u64)AUDIO_CAPTURE_RING_BASE;
    Params.active_ch = AUDIO_CHANNELS;
    Params.bits_per_sample = BIT_DEPTH_16;
    Params.periods = AUDIO_PERIOD_COUNT;
    Params.bytes_per_period = AUDIO_PERIOD_BYTES;
    XAudioFormatterSetHwParams(&AudioFormatter, &Params);

    XAudioFormatterDMAStart(&AudioFormatter);
    XI2s_Rx_Enable(&I2sReceiver, TRUE);
    XI2s_Tx_Enable(&ClockTransmitter, TRUE);
    xil_printf("[CAPTURE] Speak now: 10 s, board audio output disabled\r\n");

    for (Period = 0U; Period < AUDIO_TOTAL_PERIODS; ++Period) {
        u32 RingPeriod = Period % AUDIO_PERIOD_COUNT;
        u32 SourceBase = RingPeriod * AUDIO_PERIOD_FRAMES * AUDIO_CHANNELS;
        u32 DestBase = Period * AUDIO_PERIOD_FRAMES;
        u32 DelayUs = AUDIO_PERIOD_TIME_US;
        u32 Frame;

        if (Period == 0U) {
            DelayUs += AUDIO_FIRST_PERIOD_GUARD_US;
        }
        usleep(DelayUs);
        Status = AudioCapture_CheckStatus();
        if (Status != XST_SUCCESS) {
            break;
        }

        Xil_DCacheInvalidateRange((INTPTR)&Ring[SourceBase],
            AUDIO_PERIOD_BYTES);
        for (Frame = 0U; Frame < AUDIO_PERIOD_FRAMES; ++Frame) {
            s32 Left = Ring[SourceBase + Frame * AUDIO_CHANNELS];
            s32 Right = Ring[SourceBase + Frame * AUDIO_CHANNELS + 1U];
            NpuInput[DestBase + Frame] = (s16)((Left + Right) / 2);
        }
    }

    XI2s_Rx_Enable(&I2sReceiver, FALSE);
    XI2s_Tx_Enable(&ClockTransmitter, FALSE);
    AudioFormatter.ChannelId = XAudioFormatter_S2MM;
    XAudioFormatterDMAStop(&AudioFormatter);
    if (Status != XST_SUCCESS) {
        return XST_FAILURE;
    }

    AudioCapture_ConditionNpuInput(NpuInput);
    Xil_DCacheFlushRange((INTPTR)NpuInput, AUDIO_MONO_BYTES);
    xil_printf("[CAPTURE] Complete: %u mono samples -> NPU input 0x%08x\r\n",
        (unsigned)AUDIO_MONO_SAMPLE_COUNT, (unsigned)NPU_INPUT_BASE);
    return XST_SUCCESS;
}

void AudioCapture_PrintSummary(void)
{
    const s16 *Samples = AudioCapture_NpuInput();
    s32 MinValue = 32767;
    s32 MaxValue = -32768;
    s64 Sum = 0;
    u64 SumAbs = 0U;
    u32 Index;

    for (Index = 0U; Index < AUDIO_MONO_SAMPLE_COUNT; ++Index) {
        s32 Value = Samples[Index];
        if (Value < MinValue) MinValue = Value;
        if (Value > MaxValue) MaxValue = Value;
        Sum += Value;
        SumAbs += (Value < 0) ? (u32)(-Value) : (u32)Value;
    }
    xil_printf("[CAPTURE] mono min=%d max=%d mean=%d mean_abs=%u first=",
        (int)MinValue, (int)MaxValue,
        (int)(Sum / (s64)AUDIO_MONO_SAMPLE_COUNT),
        (unsigned)(SumAbs / AUDIO_MONO_SAMPLE_COUNT));
    for (Index = 0U; Index < 8U; ++Index) {
        xil_printf("%d%s", (int)Samples[Index], (Index == 7U) ? "\r\n" : ",");
    }
}
