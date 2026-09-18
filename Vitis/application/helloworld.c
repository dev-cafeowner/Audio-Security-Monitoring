#include "audio_capture.h"
#include "audio_playback.h"
#include "audio_stream.h"
#include "npu_inference.h"
#include "npu_memory_map.h"
#include "platform.h"
#include "ps_postprocess.h"
#include "ssm2603.h"
#include "ui_uart.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xuartps_hw.h"
#include "xstatus.h"
#include "sleep.h"

#include <string.h>

static u32 g_ui_result_sequence = 0U;

static int SendUiResult(u32 FinalOutputAddress)
{
    const s16 *Logits = (const s16 *)(UINTPTR)FinalOutputAddress;
    ui_result_t Result;

    Xil_DCacheInvalidateRange((INTPTR)FinalOutputAddress,
                              NPU_INFERENCE_RESULT_VALUES * sizeof(s16));
    if (PsPostprocess_Run(Logits, &Result) != 0) {
        xil_printf("UI_RESULT_FAILED,postprocess\r\n");
        return XST_FAILURE;
    }

    Result.seq = ++g_ui_result_sequence;
    Result.inference_us = NpuInference_GetLastRunUs();
    UiUart_SendResult(&Result);
    return XST_SUCCESS;
}

static char ReadMode(void)
{
    char Mode;

    xil_printf("\r\nSelect input mode: [m] microphone, [k] knock, [g] glass, ");
    xil_printf("[s] gunshot, [f] fire alarm, [q] quit: ");
    do {
        Mode = inbyte();
    } while ((Mode != 'm') && (Mode != 'M') &&
             (Mode != 'k') && (Mode != 'K') &&
             (Mode != 'g') && (Mode != 'G') &&
             (Mode != 's') && (Mode != 'S') &&
             (Mode != 'f') && (Mode != 'F') &&
             (Mode != 'q') && (Mode != 'Q'));
    xil_printf("%c\r\n", Mode);
    return Mode;
}

static int UartTryReadByte(char *Value)
{
    if ((Value != NULL) &&
        XUartPs_IsReceiveData(XPAR_XUARTPS_0_BASEADDR)) {
        *Value = (char)XUartPs_RecvByte(XPAR_XUARTPS_0_BASEADDR);
        return 1;
    }
    return 0;
}

static int RunContinuousMic(int *QuitRequested)
{
    u32 FinalOutputAddress;
    char Command;
    int Status;

    *QuitRequested = 0;
    if (SSM2603_InitMic() != XST_SUCCESS ||
        AudioCapture_Init() != XST_SUCCESS ||
        AudioCapture_StartContinuous() != XST_SUCCESS) {
        return XST_FAILURE;
    }
    xil_printf("MIC_STREAM_STARTED,window_sec=10,hop_sec=%u\r\n",
        (unsigned)AUDIO_MIC_WINDOW_HOP_SEC);
    xil_printf("[CAPTURE] Continuous MIC active; send x to stop.\r\n");

    while (AudioCapture_IsContinuousActive()) {
        if (UartTryReadByte(&Command)) {
            if ((Command == 'x') || (Command == 'X') ||
                (Command == 'm') || (Command == 'M') ||
                (Command == 'q') || (Command == 'Q')) {
                if ((Command == 'q') || (Command == 'Q')) {
                    *QuitRequested = 1;
                }
                xil_printf("MIC_STREAM_STOP_REQUESTED\r\n");
                break;
            }
        }

        Status = AudioCapture_ServiceContinuous();
        if (Status != XST_SUCCESS) {
            xil_printf("MIC_STREAM_FAILED,capture_service\r\n");
            AudioCapture_StopContinuous();
            return XST_FAILURE;
        }

        if (AudioCapture_HasWindow()) {
            if (AudioCapture_SnapshotWindowToNpuInput() != XST_SUCCESS) {
                xil_printf("MIC_STREAM_FAILED,snapshot\r\n");
                AudioCapture_StopContinuous();
                return XST_FAILURE;
            }
            AudioCapture_PrintSummary();
            FinalOutputAddress = 0U;
            NpuInference_SetServiceHook(AudioCapture_ServiceContinuous);
            Status = NpuInference_Run(&FinalOutputAddress);
            NpuInference_SetServiceHook(NULL);

            /* The S2MM ring kept running while inference blocked the CPU. */
            if (AudioCapture_ServiceContinuous() != XST_SUCCESS) {
                xil_printf("MIC_STREAM_FAILED,capture_overrun_after_npu\r\n");
                AudioCapture_StopContinuous();
                return XST_FAILURE;
            }
            if (Status != 0) {
                xil_printf("MIC_STREAM_FAILED,NPU inference,rc=%d\r\n", Status);
                AudioCapture_StopContinuous();
                return XST_FAILURE;
            }

            NpuInference_PrintDemoResults(FinalOutputAddress, "MIC_STREAM");
            if (SendUiResult(FinalOutputAddress) != XST_SUCCESS) {
                xil_printf("MIC_STREAM_FAILED,UI result\r\n");
                AudioCapture_StopContinuous();
                return XST_FAILURE;
            }
            xil_printf("WINDOW_COMPLETE,mode=MIC_STREAM,index=%u,hop_sec=%u\r\n",
                (unsigned)AudioCapture_GetWindowSequence(),
                (unsigned)AUDIO_MIC_WINDOW_HOP_SEC);
        }
        usleep(1000U);
    }

    AudioCapture_StopContinuous();
    xil_printf("MIC_STREAM_STOPPED,windows=%u,samples=%u\r\n",
        (unsigned)AudioCapture_GetWindowSequence(),
        (unsigned)AudioCapture_GetCapturedSamples());
    return XST_SUCCESS;
}

static int GetDemoSelection(char Mode, u32 *SourceAddress,
                            const char **ModeName)
{
    switch (Mode) {
    case 'k': case 'K':
        *SourceAddress = AUDIO_DEMO_KNOCK_BASE;
        *ModeName = "DEMO_KNOCK";
        return XST_SUCCESS;
    case 'g': case 'G':
        *SourceAddress = AUDIO_DEMO_GLASS_BASE;
        *ModeName = "DEMO_GLASS_BREAK";
        return XST_SUCCESS;
    case 's': case 'S':
        *SourceAddress = AUDIO_DEMO_GUNSHOT_BASE;
        *ModeName = "DEMO_GUNSHOT";
        return XST_SUCCESS;
    case 'f': case 'F':
        *SourceAddress = AUDIO_DEMO_FIRE_ALARM_BASE;
        *ModeName = "DEMO_FIRE_ALARM";
        return XST_SUCCESS;
    default:
        return XST_FAILURE;
    }
}

static int StartDemoPlayback(u32 SourceAddress, const char *ModeName)
{
    const s16 *DemoMono = (const s16 *)(UINTPTR)NPU_INPUT_BASE;
    const s16 *DemoSource = (const s16 *)(UINTPTR)SourceAddress;

    /* MIC capture overwrites NPU_INPUT_BASE. Restore the immutable demo copy. */
    Xil_DCacheInvalidateRange((INTPTR)DemoSource, AUDIO_MONO_BYTES);
    memcpy((void *)(UINTPTR)NPU_INPUT_BASE, DemoSource, AUDIO_MONO_BYTES);
    Xil_DCacheFlushRange((INTPTR)NPU_INPUT_BASE, AUDIO_MONO_BYTES);
    xil_printf("[DEMO] %s restored %u bytes: 0x%08x -> NPU input 0x%08x\r\n",
        ModeName, (unsigned)AUDIO_MONO_BYTES, (unsigned)SourceAddress,
        (unsigned)NPU_INPUT_BASE);

    if (SSM2603_InitPlayback() != XST_SUCCESS ||
        AudioPlayback_Init() != XST_SUCCESS ||
        AudioPlayback_StartDemo(DemoMono) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int main(void)
{
    u32 FinalOutputAddress;
    u32 DemoSourceAddress;
    const char *ModeName;
    char Mode;
    int Status;
    int QuitRequested;

    init_platform();
    UiUart_Init();
    xil_printf("\r\n================================================\r\n");
    xil_printf(" ASM MIC/DEMO -> NPU INTEGRATION (NPU IP v7.0 RELEASE)\r\n");
    xil_printf("================================================\r\n");
    xil_printf("MIC : continuous codec ADC -> S2MM; 10 s window / 5 s hop\r\n");
    xil_printf("      board audio output is disabled; send x to stop\r\n");
    xil_printf("DEMO: knock/glass/gunshot/fire DDR mono -> NPU + board audio\r\n");
    xil_printf("PC  : UART shows Top-5 plus machine-readable RESULT packets\r\n");
    xil_printf("Model/FC payloads must already be provisioned over JTAG.\r\n");

    Status = NpuInference_Init();
    if (Status != 0) {
        xil_printf("FATAL,NPU initialization failed,rc=%d\r\n", Status);
        cleanup_platform();
        return -1;
    }
    Status = NpuInference_PrepareModel();
    if (Status != 0) {
        xil_printf("FATAL,model payload map invalid,rc=%d\r\n", Status);
        cleanup_platform();
        return -1;
    }

    for (;;) {
        Mode = ReadMode();
        if ((Mode == 'q') || (Mode == 'Q')) {
            xil_printf("APPLICATION_EXIT\r\n");
            break;
        }

        FinalOutputAddress = 0U;
        if ((Mode == 'm') || (Mode == 'M')) {
            ModeName = "MIC_STREAM";
            xil_printf("MODE,%s\r\n", ModeName);
            Status = RunContinuousMic(&QuitRequested);
            if (Status != XST_SUCCESS) {
                xil_printf("RUN_FAILED,continuous MIC,rc=%d\r\n", Status);
            }
            if (QuitRequested) {
                xil_printf("APPLICATION_EXIT\r\n");
                break;
            }
            continue;
        } else {
            Status = GetDemoSelection(Mode, &DemoSourceAddress, &ModeName);
            if (Status == XST_SUCCESS) {
                xil_printf("MODE,%s\r\n", ModeName);
                Status = StartDemoPlayback(DemoSourceAddress, ModeName);
            }
        }
        if (Status != XST_SUCCESS) {
            xil_printf("RUN_FAILED,audio path setup,rc=%d\r\n", Status);
            if (AudioPlayback_IsActive()) {
                AudioPlayback_Stop();
            }
            continue;
        }

        Status = NpuInference_Run(&FinalOutputAddress);
        if (AudioPlayback_IsActive()) {
            AudioPlayback_WaitUntilDone();
        }
        if ((ModeName != NULL) && (strcmp(ModeName, "MIC") != 0) &&
            AudioPlayback_HadError()) {
            xil_printf("RUN_FAILED,demo playback DMA error\r\n");
            continue;
        }
        if (Status != 0) {
            xil_printf("RUN_FAILED,NPU inference,rc=%d\r\n", Status);
            continue;
        }

        NpuInference_PrintDemoResults(FinalOutputAddress, ModeName);
        if (SendUiResult(FinalOutputAddress) != XST_SUCCESS) {
            xil_printf("RUN_FAILED,UI result\r\n");
            continue;
        }
        xil_printf("RUN_COMPLETE,mode=%s\r\n", ModeName);
    }
    cleanup_platform();
    return 0;
}
