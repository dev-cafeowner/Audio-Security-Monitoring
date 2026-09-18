#include "audio_playback.h"

#include "audio_i2s_clock.h"
#include "audio_stream.h"
#include "xaudioformatter.h"
#include "xaudioformatter_hw.h"
#include "xi2stx.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "sleep.h"

#define AF_BASEADDR             XPAR_XAUDIO_FORMATTER_0_BASEADDR
#define I2S_TX_BASEADDR         XPAR_XI2STX_0_BASEADDR
#define AUDIO_MCLK_HZ           12288000U
#define AF_MM2S_ERROR_MASK      (BIT(17) | BIT(18) | BIT(19) | BIT(30))
#define PLAYBACK_STALL_POLLS    3000U

static XAudioFormatter AudioFormatter;
static XI2s_Tx I2sTransmitter;
static const s16 *DemoMono;
static u32 NextSourcePeriod;
static u32 LastRawTransferCount;
static u32 TotalTransferCount;
static u32 StallPolls;
static int PlaybackActive;
static int PlaybackError;

static s16 *AudioPlayback_Ring(void)
{
    return (s16 *)(UINTPTR)AUDIO_PLAYBACK_RING_BASE;
}

static void AudioPlayback_FillPeriod(u32 SourcePeriod)
{
    s16 *Ring = AudioPlayback_Ring();
    u32 RingPeriod = SourcePeriod % AUDIO_PERIOD_COUNT;
    u32 SourceBase = SourcePeriod * AUDIO_PERIOD_FRAMES;
    u32 DestBase = RingPeriod * AUDIO_PERIOD_FRAMES * AUDIO_CHANNELS;
    u32 Frame;

    for (Frame = 0U; Frame < AUDIO_PERIOD_FRAMES; ++Frame) {
        s16 Sample = DemoMono[SourceBase + Frame];
        Ring[DestBase + Frame * AUDIO_CHANNELS] = Sample;
        Ring[DestBase + Frame * AUDIO_CHANNELS + 1U] = Sample;
    }
    Xil_DCacheFlushRange(
        (INTPTR)&Ring[DestBase], AUDIO_PERIOD_BYTES);
}

int AudioPlayback_Init(void)
{
    XI2stx_Config *I2sConfig;
    int Status;

    Status = (int)XAudioFormatter_Initialize(&AudioFormatter, AF_BASEADDR);
    if ((Status != XST_SUCCESS) || !AudioFormatter.mm2s_presence) {
        xil_printf("[PLAYBACK] Audio Formatter MM2S init failed\r\n");
        return XST_FAILURE;
    }
    AudioFormatter.ChannelId = XAudioFormatter_MM2S;

    I2sConfig = XI2s_Tx_LookupConfig((UINTPTR)I2S_TX_BASEADDR);
    if (I2sConfig == NULL) {
        xil_printf("[PLAYBACK] I2S transmitter config not found\r\n");
        return XST_FAILURE;
    }
    Status = AudioI2sClock_InitializeMaster(&I2sTransmitter, I2sConfig);
    if (Status != XST_SUCCESS) {
        xil_printf("[PLAYBACK] I2S transmitter init failed (%d)\r\n", Status);
        return XST_FAILURE;
    }
    XI2s_Tx_Enable(&I2sTransmitter, FALSE);
    Status = XI2s_Tx_SetChMux(&I2sTransmitter, XI2S_TX_CHID0,
        XI2S_TX_CHMUX_AXIS_01);
    if (Status != XST_SUCCESS) {
        xil_printf("[PLAYBACK] I2S channel mux failed (%d)\r\n", Status);
        return XST_FAILURE;
    }
    if (!I2sTransmitter.Config.IsMaster ||
        !I2sTransmitter.Config.Is32BitLR) {
        xil_printf("[PLAYBACK] I2S TX must be 32-bit-slot master\r\n");
        return XST_FAILURE;
    }
    Status = (int)XI2s_Tx_SetSclkOutDiv(&I2sTransmitter,
        AUDIO_MCLK_HZ, AUDIO_SAMPLE_RATE_HZ);
    if (Status != XST_SUCCESS) {
        xil_printf("[PLAYBACK] I2S clock divider failed (%d)\r\n", Status);
        return XST_FAILURE;
    }

    PlaybackActive = 0;
    PlaybackError = 0;
    xil_printf("[PLAYBACK] MM2S=0x%08x I2S-TX=0x%08x ring=0x%08x\r\n",
        (unsigned)AF_BASEADDR, (unsigned)I2S_TX_BASEADDR,
        (unsigned)AUDIO_PLAYBACK_RING_BASE);
    return XST_SUCCESS;
}

int AudioPlayback_StartDemo(const s16 *MonoSamples)
{
    XAudioFormatterHwParams Params;
    u32 Period;

    if (MonoSamples == NULL) {
        return XST_FAILURE;
    }
    DemoMono = MonoSamples;
    Xil_DCacheInvalidateRange((INTPTR)DemoMono, AUDIO_MONO_BYTES);

    for (Period = 0U; Period < AUDIO_PERIOD_COUNT; ++Period) {
        AudioPlayback_FillPeriod(Period);
    }
    NextSourcePeriod = AUDIO_PERIOD_COUNT;
    LastRawTransferCount = 0U;
    TotalTransferCount = 0U;
    StallPolls = 0U;
    PlaybackError = 0;

    AudioFormatter.ChannelId = XAudioFormatter_MM2S;
    XAudioFormatterDMAStop(&AudioFormatter);
    XAudioFormatterDMAReset(&AudioFormatter);
    Params.buf_addr = (u64)AUDIO_PLAYBACK_RING_BASE;
    Params.active_ch = AUDIO_CHANNELS;
    Params.bits_per_sample = BIT_DEPTH_16;
    Params.periods = AUDIO_PERIOD_COUNT;
    Params.bytes_per_period = AUDIO_PERIOD_BYTES;
    XAudioFormatterSetFsMultiplier(&AudioFormatter,
        AUDIO_MCLK_HZ, AUDIO_SAMPLE_RATE_HZ);
    XAudioFormatterSetHwParams(&AudioFormatter, &Params);

    /* The verified sequence establishes BCLK/LRCLK before starting MM2S. */
    XI2s_Tx_Enable(&I2sTransmitter, TRUE);
    usleep(100U);
    XAudioFormatterDMAStart(&AudioFormatter);
    PlaybackActive = 1;
    xil_printf("[PLAYBACK] DEMO start: 32 kHz PCM16 mono->stereo, 10 s\r\n");
    return XST_SUCCESS;
}

void AudioPlayback_Service(void)
{
    u32 Status;
    u32 RawTransferCount;
    u32 Delta;
    u32 CompletedPeriods;

    if (!PlaybackActive) {
        return;
    }
    AudioFormatter.ChannelId = XAudioFormatter_MM2S;
    Status = XAudioFormatter_ReadReg(AF_BASEADDR,
        XAUD_FORMATTER_MM2S_OFFSET + XAUD_FORMATTER_STS);
    if ((Status & AF_MM2S_ERROR_MASK) != 0U) {
        xil_printf("[PLAYBACK] MM2S error status=0x%08x\r\n",
            (unsigned)Status);
        PlaybackError = 1;
        AudioPlayback_Stop();
        return;
    }

    /* XFER_COUNT is the position in the cyclic ring, not a lifetime count. */
    RawTransferCount = XAudioFormatterGetDMATransferCount(&AudioFormatter);
    if (RawTransferCount >= LastRawTransferCount) {
        Delta = RawTransferCount - LastRawTransferCount;
    } else {
        Delta = AUDIO_RING_BYTES - LastRawTransferCount + RawTransferCount;
    }
    LastRawTransferCount = RawTransferCount;
    TotalTransferCount += Delta;
    CompletedPeriods = TotalTransferCount / AUDIO_PERIOD_BYTES;

    if (Delta != 0U) {
        StallPolls = 0U;
    } else if (++StallPolls >= PLAYBACK_STALL_POLLS) {
        xil_printf("[PLAYBACK] DMA stalled: total=%u raw=%u status=0x%08x\r\n",
            (unsigned)TotalTransferCount, (unsigned)RawTransferCount,
            (unsigned)Status);
        PlaybackError = 1;
        AudioPlayback_Stop();
        return;
    }

    /* A ring slot is safe to refill immediately after its old period ends. */
    while ((NextSourcePeriod < AUDIO_TOTAL_PERIODS) &&
           (NextSourcePeriod < (CompletedPeriods + AUDIO_PERIOD_COUNT))) {
        AudioPlayback_FillPeriod(NextSourcePeriod);
        ++NextSourcePeriod;
    }

    if (TotalTransferCount >= AUDIO_STEREO_BYTES) {
        AudioPlayback_Stop();
        xil_printf("[PLAYBACK] DEMO complete: %u bytes\r\n",
            (unsigned)AUDIO_STEREO_BYTES);
    }
}

int AudioPlayback_IsActive(void)
{
    return PlaybackActive;
}

int AudioPlayback_HadError(void)
{
    return PlaybackError;
}

void AudioPlayback_WaitUntilDone(void)
{
    while (PlaybackActive) {
        AudioPlayback_Service();
        usleep(1000U);
    }
}

void AudioPlayback_Stop(void)
{
    AudioFormatter.ChannelId = XAudioFormatter_MM2S;
    XI2s_Tx_Enable(&I2sTransmitter, FALSE);
    XAudioFormatterDMAStop(&AudioFormatter);
    PlaybackActive = 0;
}
