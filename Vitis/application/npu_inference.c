/*
 * Unified NPU hardware benchmark and full-model Golden gate.
 *
 * BENCH_RUN_MODEL_GOLDEN_CHAIN runs the canonical model_params.bin and
 * 00_input_int16_q15.bin through Conv1..Conv9, GLOBAL, FC1, and FC2 without
 * an inter-layer reset, then compares all 527 final INT16 values against the
 * B-handoff FC2 Golden output.  The synthetic timing modes remain available
 * through the other BENCH_* switches.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "platform.h"
#include "audio_playback.h"
#include "npu_class_labels.h"
#include "npu_inference.h"
#include "npu_unified_driver.h"
#include "sleep.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include "xparameters.h"
#include "xtimer_config.h"
#include "npu_memory_map.h"

#ifndef BENCH_LAYER_MASK
#define BENCH_LAYER_MASK          0x0FFFU
#endif

static NpuInferenceServiceHook g_service_hook;
#ifndef BENCH_WARMUP_RUNS
#define BENCH_WARMUP_RUNS         1U
#endif
#ifndef BENCH_MEASURED_RUNS
#define BENCH_MEASURED_RUNS       3U
#endif
#ifndef BENCH_TIMEOUT_SECONDS
#define BENCH_TIMEOUT_SECONDS     120U
#endif
#ifndef BENCH_PL_CLOCK_HZ
#define BENCH_PL_CLOCK_HZ          76923080U
#endif
#ifndef BENCH_CONVN_CIN_PER_STEP
#define BENCH_CONVN_CIN_PER_STEP          4U
#endif
#ifndef BENCH_SOFT_RESET_EACH_RUN
#define BENCH_SOFT_RESET_EACH_RUN  1U
#endif
#ifndef BENCH_RUN_NO_RESET_CHAIN
#define BENCH_RUN_NO_RESET_CHAIN    1U
#endif
#ifndef BENCH_RUN_TRANSITION_SWEEP
#define BENCH_RUN_TRANSITION_SWEEP  0U
#endif
#ifndef BENCH_RUN_COLD_BENCHMARK
#define BENCH_RUN_COLD_BENCHMARK    0U
#endif
#ifndef BENCH_RUN_MODEL_GOLDEN_CHAIN
#define BENCH_RUN_MODEL_GOLDEN_CHAIN 1U
#endif
#ifndef BENCH_RUN_CONV2_STANDALONE_DIAG
/*
 * Diagnostic-only mode: consume the exact Conv1 output already resident in
 * ACT_A, soft-reset the unified NPU, and run Conv2 alone into ACT_B.  This
 * separates a Conv1->Conv2 no-reset transition defect from a ConvN datapath or
 * shared-parameter defect.  Do not re-provision or overwrite ACT_A before use.
 */
#define BENCH_RUN_CONV2_STANDALONE_DIAG 0U
#endif
#ifndef BENCH_CHAIN_TIMEOUT_SECONDS
#define BENCH_CHAIN_TIMEOUT_SECONDS 5U
#endif

#define ALIGN64(v)                (((v) + 63U) & ~63U)
#define DMA_MAX_TRANSFER_BYTES    0x00FFFFFFU

/* Host-provisioned, verified FC tile images and final Golden output. */
#define NPU_FC1_TILE_BASE         (NPU_GOLDEN_BASE + 0x01800000U)
#define NPU_FC2_TILE_BASE         (NPU_GOLDEN_BASE + 0x01900000U)
#define NPU_FC1_TILE_BYTES        264192U
#define NPU_FC2_TILE_BYTES        541728U
#define NPU_FC_STREAM_BASE        NPU_OUTPUT_BASE
#define NPU_FC_STREAM_REGION_SIZE NPU_OUTPUT_REGION_SIZE
#define NPU_FINAL_GOLDEN_VALUES   527U
#define NPU_FINAL_GOLDEN_BYTES    (NPU_FINAL_GOLDEN_VALUES * 2U)
#define NPU_DEMO_TOP_K            5U

#ifndef NPU_UART_PRINT_RAW_RESULTS
#define NPU_UART_PRINT_RAW_RESULTS 0U
#endif

static u32 g_last_run_us = 0U;

typedef enum {
    LAYER_CONV,
    LAYER_GLOBAL,
    LAYER_FC
} layer_kind_t;

typedef struct {
    u8 id;
    const char *name;
    layer_kind_t kind;
    u32 cin;
    u32 cout;
    u32 groups;
    u32 tconv;
    u32 tout;
    u32 input_bytes;
    u32 output_bytes;
    u32 fc_chunks;
} layer_desc_t;

typedef struct {
    u32 weight_ofs;
    u32 weight_bytes;
    u32 bias_ofs;
    u32 bias_bytes;
    u32 rshift_ofs;
    u32 rshift_bytes;
    u32 fc_tile_addr;
    u32 fc_tile_bytes;
} layer_param_desc_t;

typedef struct {
    XTime setup_ticks;
    XTime load_ticks;
    XTime total_ticks;
    XTime tx_done_ticks;
    XTime rx_done_ticks;
    XTime npu_done_ticks;
    u32 traffic_bytes;
} bench_sample_t;

typedef struct {
    XTime total_min;
    XTime total_max;
    XTime total_sum;
    XTime setup_sum;
    XTime load_sum;
    u32 count;
    u32 traffic_bytes;
} bench_stats_t;

static const layer_desc_t g_layers[] = {
    { 0U, "Conv1",  LAYER_CONV,     1U,  64U,  4U, 106667U, 106667U,   640000U, 13653376U,   0U },
    { 1U, "Conv2",  LAYER_CONV,    64U,  64U,  4U, 106667U,  35556U, 13653376U,  4551168U,   0U },
    { 2U, "Conv3",  LAYER_CONV,    64U,  64U,  4U,  35556U,  11852U,  4551168U,  1517056U,   0U },
    { 3U, "Conv4",  LAYER_CONV,    64U, 128U,  8U,  11852U,   3951U,  1517056U,  1011456U,   0U },
    { 4U, "Conv5",  LAYER_CONV,   128U, 128U,  8U,   3951U,   1317U,  1011456U,   337152U,   0U },
    { 5U, "Conv6",  LAYER_CONV,   128U, 128U,  8U,   1317U,    439U,   337152U,   112384U,   0U },
    { 6U, "Conv7",  LAYER_CONV,   128U, 128U,  8U,    439U,    147U,   112384U,    37632U,   0U },
    { 7U, "Conv8",  LAYER_CONV,   128U, 128U,  8U,    147U,     49U,    37632U,    12544U,   0U },
    { 8U, "Conv9",  LAYER_CONV,   128U, 256U, 16U,     49U,     17U,    12544U,     8704U,   0U },
    { 9U, "GLOBAL", LAYER_GLOBAL, 256U, 256U, 16U,     17U,      1U,     8704U,      512U,   0U },
    {10U, "FC1",    LAYER_FC,     256U, 512U, 32U,      0U,      0U,      512U,     1024U,  86U },
    {11U, "FC2",    LAYER_FC,     512U, 527U, 33U,      0U,      0U,     1024U,     1054U, 171U }
};

/* model_manifest.json canonical offsets; all offsets are NPU_MODEL_BASE-relative. */
static const layer_param_desc_t g_layer_params[] = {
    {      0U,    384U,     384U,  512U,     896U,  64U, 0U, 0U },
    {    960U,  24576U,   25536U,  512U,   26048U,  64U, 0U, 0U },
    {  26112U,  24576U,   50688U,  512U,   51200U,  64U, 0U, 0U },
    {  51264U,  49152U,  100416U, 1024U,  101440U, 128U, 0U, 0U },
    { 101568U,  98304U,  199872U, 1024U,  200896U, 128U, 0U, 0U },
    { 201024U,  98304U,  299328U, 1024U,  300352U, 128U, 0U, 0U },
    { 300480U,  98304U,  398784U, 1024U,  399808U, 128U, 0U, 0U },
    { 399936U,  98304U,  498240U, 1024U,  499264U, 128U, 0U, 0U },
    { 499392U, 196608U,  696000U, 2048U,  698048U, 256U, 0U, 0U },
    {      0U,      0U,       0U,    0U,       0U,   0U, 0U, 0U },
    { 698304U, 262144U,  960448U, 4096U,  964544U, 512U,
      NPU_FC1_TILE_BASE, NPU_FC1_TILE_BYTES },
    { 965056U, 539648U, 1504704U, 4216U, 1508928U, 527U,
      NPU_FC2_TILE_BASE, NPU_FC2_TILE_BYTES }
};

/* 64-byte-aligned locations of all 12 host-provisioned Golden tensors. */
static const u32 g_layer_golden_addr[] = {
    NPU_GOLDEN_BASE + 0x00000000U,
    NPU_GOLDEN_BASE + 0x00D05580U,
    NPU_GOLDEN_BASE + 0x0115C780U,
    NPU_GOLDEN_BASE + 0x012CED80U,
    NPU_GOLDEN_BASE + 0x013C5C80U,
    NPU_GOLDEN_BASE + 0x01418180U,
    NPU_GOLDEN_BASE + 0x01433880U,
    NPU_GOLDEN_BASE + 0x0143CB80U,
    NPU_GOLDEN_BASE + 0x0143FC80U,
    NPU_GOLDEN_BASE + 0x01441E80U,
    NPU_GOLDEN_BASE + 0x01442080U,
    NPU_GOLDEN_BASE + 0x01442480U
};

static XAxiDma g_dma;

/*
 * Vitis 2024.2 xiltimer's Zynq default backend installs its function
 * pointers in a constructor, but starts the PS global timer lazily on the
 * first XilTimer_Sleep() call.  Reading XTime_GetTime() before that returns
 * a permanent zero on this platform.  Start it explicitly and reject a
 * non-advancing timer so a broken measurement can never look like a valid
 * zero-latency benchmark.
 */
static int init_benchmark_timer(XTime *self_test_delta)
{
    XTime before;
    XTime after;

    if (XilSleepTimer_Init(&TimerInst) != XST_SUCCESS) {
        return -1;
    }
    usleep(1U);
    XTime_GetTime(&before);
    usleep(1U);
    XTime_GetTime(&after);
    if (self_test_delta != NULL) {
        *self_test_delta = after - before;
    }
    return (after > before) ? 0 : -2;
}

static XTime timer_now(void)
{
    XTime now;
    XTime_GetTime(&now);
    return now;
}

/*
 * xtimer_config.h defines COUNTS_PER_SECOND through an unparenthesized
 * CPU_CLOCK/2 macro on Zynq-7000.  Using that macro directly as the right
 * operand of / or % changes the expression grouping after preprocessing.
 * Materialize it once behind parentheses and use the resulting value for all
 * benchmark arithmetic.
 */
static uint64_t benchmark_timer_hz(void)
{
    return (uint64_t)(COUNTS_PER_SECOND);
}

static uint64_t ticks_to_ns(XTime ticks)
{
    const uint64_t timer_hz = benchmark_timer_hz();
    uint64_t whole = (uint64_t)ticks / timer_hz;
    uint64_t rem = (uint64_t)ticks % timer_hz;
    return whole * 1000000000ULL +
           (rem * 1000000000ULL) / timer_hz;
}

static uint64_t ticks_to_pl_cycles(XTime ticks)
{
    const uint64_t timer_hz = benchmark_timer_hz();
    uint64_t whole = (uint64_t)ticks / timer_hz;
    uint64_t rem = (uint64_t)ticks % timer_hz;
    return whole * (uint64_t)BENCH_PL_CLOCK_HZ +
           (rem * (uint64_t)BENCH_PL_CLOCK_HZ) / timer_hz;
}

static void print_time_us(XTime ticks)
{
    uint64_t ns = ticks_to_ns(ticks);
    printf("%llu.%03llu", (unsigned long long)(ns / 1000ULL),
           (unsigned long long)(ns % 1000ULL));
}

static void dma_reset(void)
{
    u32 poll;

    XAxiDma_Reset(&g_dma);
    for (poll = 0U; poll < 1000000U; ++poll) {
        AudioPlayback_Service();
        if (XAxiDma_ResetIsDone(&g_dma)) {
            break;
        }
    }
}

static u32 dma_status(int direction)
{
    u32 channel_offset = (direction == XAXIDMA_DMA_TO_DEVICE) ?
                         XAXIDMA_TX_OFFSET : XAXIDMA_RX_OFFSET;
    return XAxiDma_ReadReg(g_dma.RegBase + channel_offset, XAXIDMA_SR_OFFSET);
}

static int dma_channel_ok(int direction)
{
    return (dma_status(direction) & XAXIDMA_ERR_ALL_MASK) == 0U;
}

static int dma_wait_idle(const char *stage, int direction)
{
    XTime start = timer_now();
    XTime timeout = (XTime)benchmark_timer_hz() * BENCH_TIMEOUT_SECONDS;
    int diagnostic_printed = 0;

    while (XAxiDma_Busy(&g_dma, direction)) {
        XTime elapsed = timer_now() - start;
        AudioPlayback_Service();
        if (!diagnostic_printed &&
            (elapsed > (XTime)benchmark_timer_hz())) {
            npu_unified_status_t status;
            npu_unified_read_status(&status);
            printf("DIAG: stalled >1s at %s status=0x%08lx TXSR=0x%08lx RXSR=0x%08lx\r\n",
                   stage, (unsigned long)status.raw,
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
            diagnostic_printed = 1;
        }
        if (elapsed > timeout) {
            return -1;
        }
    }
    return dma_channel_ok(direction) ? 0 : -2;
}

static int dma_tx_blocking(const char *stage, UINTPTR addr, u32 bytes)
{
    if ((bytes == 0U) || (bytes > DMA_MAX_TRANSFER_BYTES)) {
        return -1;
    }
    if (XAxiDma_SimpleTransfer(&g_dma, addr, bytes,
                               XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
        return -2;
    }
    return dma_wait_idle(stage, XAXIDMA_DMA_TO_DEVICE);
}

static u32 conv_weight_bytes(const layer_desc_t *layer)
{
    return 3U * layer->cin * layer->cout * 2U;
}

static u32 conv_bias_bytes(const layer_desc_t *layer)
{
    return layer->cout * 8U;
}

static u32 conv_rshift_bytes(const layer_desc_t *layer)
{
    return layer->groups * 16U;
}

static u32 fc_stream_bytes(const layer_desc_t *layer)
{
    u32 activation = layer->cin * 2U;
    u32 bias = layer->groups * 32U * 4U;
    u32 rshift = layer->groups * 4U * 4U;
    u32 tiles = layer->groups * layer->fc_chunks * 24U * 4U;
    return activation + bias + rshift + tiles;
}

static u32 layer_tx_bytes(const layer_desc_t *layer)
{
    if (layer->kind == LAYER_CONV) {
        return conv_weight_bytes(layer) + conv_bias_bytes(layer) +
               conv_rshift_bytes(layer) + layer->input_bytes;
    }
    if (layer->kind == LAYER_FC) {
        return fc_stream_bytes(layer);
    }
    return layer->input_bytes;
}

static int wait_operation(XTime start, u32 timeout_seconds,
                          bench_sample_t *sample)
{
    XTime timeout = (XTime)benchmark_timer_hz() * timeout_seconds;
    XTime now;
    int tx_seen = 0;
    int rx_seen = 0;
    int npu_seen = 0;
    int diagnostic_printed = 0;
    npu_unified_status_t status;

    while (1) {
        now = timer_now();
        AudioPlayback_Service();
        npu_unified_read_status(&status);
        if (status.error) {
            printf("FAULT: NPU error code=0x%02x status=0x%08lx\r\n",
                   status.error_code, (unsigned long)status.raw);
            return -1;
        }

        if (!tx_seen && !XAxiDma_Busy(&g_dma, XAXIDMA_DMA_TO_DEVICE)) {
            sample->tx_done_ticks = now - start;
            tx_seen = 1;
        }
        if (!rx_seen && !XAxiDma_Busy(&g_dma, XAXIDMA_DEVICE_TO_DMA)) {
            sample->rx_done_ticks = now - start;
            rx_seen = 1;
        }
        if (!npu_seen && status.done) {
            sample->npu_done_ticks = now - start;
            npu_seen = 1;
        }

        if (tx_seen && rx_seen && npu_seen && !status.busy) {
            sample->total_ticks = now - start;
            break;
        }
        if (!diagnostic_printed &&
            ((now - start) > (XTime)benchmark_timer_hz())) {
            printf("DIAG: operation stalled >1s status=0x%08lx TXSR=0x%08lx RXSR=0x%08lx seen(tx/rx/npu)=%d/%d/%d\r\n",
                   (unsigned long)status.raw,
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA),
                   tx_seen, rx_seen, npu_seen);
            diagnostic_printed = 1;
        }
        if ((now - start) > timeout) {
            printf("FAULT: operation timeout status=0x%08lx TXSR=0x%08lx RXSR=0x%08lx\r\n",
                   (unsigned long)status.raw,
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
            return -2;
        }
    }

    if (!dma_channel_ok(XAXIDMA_DMA_TO_DEVICE) ||
        !dma_channel_ok(XAXIDMA_DEVICE_TO_DMA)) {
        printf("FAULT: DMA error TXSR=0x%08lx RXSR=0x%08lx\r\n",
               (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
               (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
        return -3;
    }
    npu_unified_clear_status();
    return 0;
}

static int run_layer_once(const layer_desc_t *layer,
                          u32 input_addr, u32 output_addr,
                          int apply_soft_reset, u32 timeout_seconds,
                          bench_sample_t *sample)
{
    const layer_param_desc_t *params = &g_layer_params[layer->id];
    u32 weight_addr = NPU_MODEL_BASE + params->weight_ofs;
    u32 weight_bytes = 0U;
    u32 bias_addr = NPU_MODEL_BASE + params->bias_ofs;
    u32 bias_bytes = 0U;
    u32 rshift_addr = NPU_MODEL_BASE + params->rshift_ofs;
    u32 rshift_bytes = 0U;
    u32 tx_bytes;
    XTime setup_start;
    XTime start;
    int rc;

    memset(sample, 0, sizeof(*sample));
    tx_bytes = layer_tx_bytes(layer);
    sample->traffic_bytes = tx_bytes + layer->output_bytes;

    if ((layer->input_bytes > NPU_ACT_REGION_SIZE) ||
        (layer->output_bytes > NPU_ACT_REGION_SIZE)) {
        return -10;
    }

    /*
     * Use a clean architectural starting state for each independent timing
     * sample. This reset is deliberately outside setup_start/CTRL.START and
     * therefore outside every reported metric. The no-reset same-layer
     * re-entry behavior is a separate RTL regression: Conv3 has been observed
     * to deadlock on its second invocation without this reset.
     */
    if (apply_soft_reset != 0) {
        rc = npu_unified_soft_reset();
        if (rc != 0) {
            return -20;
        }
    }
    XAxiDma_IntrAckIrq(&g_dma, XAXIDMA_IRQ_ALL_MASK,
                       XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrAckIrq(&g_dma, XAXIDMA_IRQ_ALL_MASK,
                       XAXIDMA_DEVICE_TO_DMA);
    /* S2MM destination pre-clean is required for coherency, but it is not
     * part of the hardware operation timing. */
    Xil_DCacheFlushRange((INTPTR)output_addr, layer->output_bytes);

    setup_start = timer_now();
    rc = npu_unified_configure(layer->id, NPU_MODEL_BASE,
                               input_addr, output_addr);
    if (rc != 0) {
        printf("FAULT: configure rc=%d\r\n", rc);
        return -11;
    }
    if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)output_addr,
                               layer->output_bytes,
                               XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
        return -12;
    }

    start = timer_now();
    sample->setup_ticks = start - setup_start;
    rc = npu_unified_start();
    if (rc != 0) {
        printf("FAULT: START rc=%d\r\n", rc);
        return -13;
    }

    if (layer->kind == LAYER_CONV) {
        weight_bytes = params->weight_bytes;
        bias_bytes = params->bias_bytes;
        rshift_bytes = params->rshift_bytes;
        if ((weight_bytes != conv_weight_bytes(layer)) ||
            (bias_bytes != conv_bias_bytes(layer)) ||
            (rshift_bytes != conv_rshift_bytes(layer))) {
            printf("FAULT: %s manifest size mismatch\r\n", layer->name);
            return -14;
        }
        if ((rshift_addr + rshift_bytes) >
            (NPU_MODEL_BASE + NPU_MODEL_REGION_SIZE)) {
            return -14;
        }
        if (dma_tx_blocking("conv weight MM2S", weight_addr, weight_bytes) != 0 ||
            dma_tx_blocking("conv bias MM2S", bias_addr, bias_bytes) != 0 ||
            dma_tx_blocking("conv rshift MM2S", rshift_addr, rshift_bytes) != 0) {
            return -15;
        }
        sample->load_ticks = timer_now() - start;
        if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)input_addr,
                                   layer->input_bytes,
                                   XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
            return -16;
        }
    } else {
        sample->load_ticks = 0U;
        if (layer->kind == LAYER_FC) {
            if (tx_bytes > NPU_FC_STREAM_REGION_SIZE) {
                return -17;
            }
            if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)NPU_FC_STREAM_BASE,
                                        tx_bytes,
                                        XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
                return -18;
            }
        } else if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)input_addr,
                                          layer->input_bytes,
                                          XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
            return -19;
        }
    }

    rc = wait_operation(start, timeout_seconds, sample);
    return rc;
}

/*
 * FC frontends consume activation, bias, rshift, and weight tiles as one
 * continuous MM2S frame. For the no-reset chain, copy the preceding layer's
 * actual S2MM output into the activation prefix of that frame.  Use the
 * dedicated NPU_OUTPUT scratch slot so FC packing cannot overwrite the
 * canonical Conv1 input at NPU_INPUT_BASE.  The synthetic benchmark keeps the
 * remaining parameter bytes zero-filled.
 */
static int prepare_fc_chain_stream(const layer_desc_t *layer, u32 input_addr,
                                   int use_real_model)
{
    const layer_param_desc_t *params = &g_layer_params[layer->id];
    u32 tx_bytes = layer_tx_bytes(layer);
    u32 bias_runtime_bytes = layer->groups * 32U * 4U;
    u32 rshift_runtime_bytes = layer->groups * 4U * 4U;
    u32 bias_dst_ofs = layer->input_bytes;
    u32 rshift_dst_ofs = bias_dst_ofs + bias_runtime_bytes;
    u32 tile_dst_ofs = rshift_dst_ofs + rshift_runtime_bytes;
    u8 *frame = (u8 *)(UINTPTR)NPU_FC_STREAM_BASE;

    if ((layer->kind != LAYER_FC) ||
        (layer->input_bytes > tx_bytes) ||
        (tx_bytes > NPU_FC_STREAM_REGION_SIZE) ||
        ((tile_dst_ofs + params->fc_tile_bytes) != tx_bytes)) {
        return -1;
    }
    Xil_DCacheInvalidateRange((INTPTR)input_addr, layer->input_bytes);
    memset(frame, 0, tx_bytes);
    memcpy(frame, (const void *)(UINTPTR)input_addr, layer->input_bytes);

    if (use_real_model != 0) {
        if ((params->bias_bytes > bias_runtime_bytes) ||
            (params->rshift_bytes > rshift_runtime_bytes) ||
            (params->fc_tile_addr < NPU_GOLDEN_BASE) ||
            ((params->fc_tile_addr + params->fc_tile_bytes) >
             (NPU_GOLDEN_BASE + NPU_GOLDEN_REGION_SIZE))) {
            return -2;
        }
        Xil_DCacheInvalidateRange((INTPTR)(NPU_MODEL_BASE + params->bias_ofs),
                                  params->bias_bytes);
        Xil_DCacheInvalidateRange((INTPTR)(NPU_MODEL_BASE + params->rshift_ofs),
                                  params->rshift_bytes);
        Xil_DCacheInvalidateRange((INTPTR)params->fc_tile_addr,
                                  params->fc_tile_bytes);
        memcpy(frame + bias_dst_ofs,
               (const void *)(UINTPTR)(NPU_MODEL_BASE + params->bias_ofs),
               params->bias_bytes);
        memcpy(frame + rshift_dst_ofs,
               (const void *)(UINTPTR)(NPU_MODEL_BASE + params->rshift_ofs),
               params->rshift_bytes);
        memcpy(frame + tile_dst_ofs,
               (const void *)(UINTPTR)params->fc_tile_addr,
               params->fc_tile_bytes);
    }
    Xil_DCacheFlushRange((INTPTR)NPU_FC_STREAM_BASE, tx_bytes);
    return 0;
}

static void stats_init(bench_stats_t *stats, u32 traffic_bytes)
{
    memset(stats, 0, sizeof(*stats));
    stats->total_min = (XTime)ULLONG_MAX;
    stats->traffic_bytes = traffic_bytes;
}

static void stats_add(bench_stats_t *stats, const bench_sample_t *sample)
{
    if (sample->total_ticks < stats->total_min) {
        stats->total_min = sample->total_ticks;
    }
    if (sample->total_ticks > stats->total_max) {
        stats->total_max = sample->total_ticks;
    }
    stats->total_sum += sample->total_ticks;
    stats->setup_sum += sample->setup_ticks;
    stats->load_sum += sample->load_ticks;
    stats->count++;
}

static void print_sample(const layer_desc_t *layer, u32 iteration,
                         const bench_sample_t *sample, int warmup)
{
    printf("%s %s%lu: total=", layer->name,
           warmup ? "warmup" : "run", (unsigned long)iteration);
    print_time_us(sample->total_ticks);
    printf(" us setup=");
    print_time_us(sample->setup_ticks);
    printf(" us load=");
    print_time_us(sample->load_ticks);
    printf(" us tx_done=");
    print_time_us(sample->tx_done_ticks);
    printf(" us rx_done=");
    print_time_us(sample->rx_done_ticks);
    printf(" us npu_done=");
    print_time_us(sample->npu_done_ticks);
    printf(" us\r\n");
}

/*
 * Host-derived ConvN throughput telemetry.  active_ticks begins immediately
 * after weight/bias/rshift loading and ends when the NPU reports DONE, so it
 * includes activation MM2S backpressure and compute but excludes parameter
 * loading.  A v6 ConvN step consumes up to four adjacent Cin values.  It is
 * intentionally not labelled as an internal RTL event count; exact
 * step_fire/core_dequeue_fire counters would require new hardware CSRs.
 */
static void print_convn_profile(const layer_desc_t *layer,
                                const bench_sample_t *sample)
{
    uint64_t expected_steps;
    XTime active_ticks;
    uint64_t active_pl_cycles;
    uint64_t cycles_per_step_milli;

    if ((layer->id < 1U) || (layer->id > 8U) ||
        (layer->kind != LAYER_CONV)) {
        return;
    }
    expected_steps =
        (((uint64_t)layer->cin + BENCH_CONVN_CIN_PER_STEP - 1ULL) /
         BENCH_CONVN_CIN_PER_STEP) *
        (uint64_t)layer->groups * (uint64_t)layer->tconv;
    active_ticks = (sample->npu_done_ticks >= sample->load_ticks) ?
                   (sample->npu_done_ticks - sample->load_ticks) : 0U;
    active_pl_cycles = ticks_to_pl_cycles(active_ticks);
    cycles_per_step_milli = (expected_steps != 0ULL) ?
        ((active_pl_cycles * 1000ULL) / expected_steps) : 0ULL;

    printf("CONVN_PROFILE,id=%lu,name=%s,expected_steps=%llu,"
           "active_pl_cycles=%llu,cycles_per_step=%llu.%03llu,"
           "cin_per_step=%lu\r\n",
           (unsigned long)layer->id, layer->name,
           (unsigned long long)expected_steps,
           (unsigned long long)active_pl_cycles,
           (unsigned long long)(cycles_per_step_milli / 1000ULL),
           (unsigned long long)(cycles_per_step_milli % 1000ULL),
           (unsigned long)BENCH_CONVN_CIN_PER_STEP);
}

static void print_summary(const layer_desc_t *layer, const bench_stats_t *stats)
{
    XTime average = stats->total_sum / stats->count;
    XTime setup_average = stats->setup_sum / stats->count;
    XTime load_average = stats->load_sum / stats->count;
    uint64_t bytes_per_second;

    if (average == 0U) {
        printf("FAULT: zero timer average for %s; summary rejected\r\n",
               layer->name);
        return;
    }
    bytes_per_second =
        ((uint64_t)stats->traffic_bytes * benchmark_timer_hz()) /
        (uint64_t)average;

    printf("SUMMARY,%lu,%s,%s,%lu,%lu,",
           (unsigned long)layer->id, layer->name,
           (layer->kind == LAYER_CONV) ? "CONV" :
           (layer->kind == LAYER_GLOBAL) ? "GLOBAL" : "FC",
           (unsigned long)layer_tx_bytes(layer),
           (unsigned long)layer->output_bytes);
    print_time_us(stats->total_min);
    printf(",");
    print_time_us(average);
    printf(",");
    print_time_us(stats->total_max);
    printf(",");
    print_time_us(setup_average);
    printf(",");
    print_time_us(load_average);
    printf(",%llu,%llu.%03llu\r\n",
           (unsigned long long)ticks_to_pl_cycles(average),
           (unsigned long long)(bytes_per_second / 1000000ULL),
           (unsigned long long)((bytes_per_second % 1000000ULL) / 1000ULL));
}

static int init_dma(void)
{
    XAxiDma_Config *config = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_BASEADDR);

    if (config == NULL) {
        return -1;
    }
    if (XAxiDma_CfgInitialize(&g_dma, config) != XST_SUCCESS) {
        return -2;
    }
    if (XAxiDma_HasSg(&g_dma)) {
        return -3;
    }
    XAxiDma_IntrDisable(&g_dma, XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&g_dma, XAXIDMA_IRQ_ALL_MASK,
                        XAXIDMA_DEVICE_TO_DMA);
    return 0;
}

static void prepare_zero_payloads(void)
{
    printf("Preparing synthetic zero payloads in DDR (outside timed region)...\r\n");
    memset((void *)(UINTPTR)NPU_MODEL_BASE, 0, NPU_MODEL_REGION_SIZE);
    memset((void *)(UINTPTR)NPU_INPUT_BASE, 0, NPU_INPUT_BYTES);
    memset((void *)(UINTPTR)NPU_ACT_A_BASE, 0, NPU_ACT_MAX_TENSOR_BYTES);
    memset((void *)(UINTPTR)NPU_ACT_B_BASE, 0, NPU_ACT_MAX_TENSOR_BYTES);
    Xil_DCacheFlushRange((INTPTR)NPU_MODEL_BASE, NPU_MODEL_REGION_SIZE);
    Xil_DCacheFlushRange((INTPTR)NPU_INPUT_BASE, NPU_INPUT_BYTES);
    Xil_DCacheFlushRange((INTPTR)NPU_ACT_A_BASE, NPU_ACT_MAX_TENSOR_BYTES);
    Xil_DCacheFlushRange((INTPTR)NPU_ACT_B_BASE, NPU_ACT_MAX_TENSOR_BYTES);
}

static int prepare_real_model_payloads(void)
{
    if ((NPU_MODEL_PARAMS_BYTES > NPU_MODEL_REGION_SIZE) ||
        (NPU_INPUT_BYTES > NPU_INPUT_REGION_SIZE) ||
        (NPU_FINAL_GOLDEN_BYTES > NPU_GOLDEN_REGION_SIZE) ||
        ((NPU_FC2_TILE_BASE + NPU_FC2_TILE_BYTES) >
         (NPU_GOLDEN_BASE + NPU_GOLDEN_REGION_SIZE))) {
        return -1;
    }

    /* XSCT/JTAG owns these bytes.  Invalidate, never flush, before use. */
    Xil_DCacheInvalidateRange((INTPTR)NPU_MODEL_BASE, NPU_MODEL_PARAMS_BYTES);
    Xil_DCacheInvalidateRange((INTPTR)NPU_INPUT_BASE, NPU_INPUT_BYTES);
    Xil_DCacheInvalidateRange((INTPTR)NPU_GOLDEN_BASE,
                              NPU_GOLDEN_REGION_SIZE);
    printf("REAL_PAYLOADS: model=0x%08lx/%lu input=0x%08lx/%lu "
           "golden=0x%08lx/%lu fc1_tile=0x%08lx/%lu fc2_tile=0x%08lx/%lu\r\n",
           (unsigned long)NPU_MODEL_BASE,
           (unsigned long)NPU_MODEL_PARAMS_BYTES,
           (unsigned long)NPU_INPUT_BASE,
           (unsigned long)NPU_INPUT_BYTES,
           (unsigned long)NPU_GOLDEN_BASE,
           (unsigned long)NPU_FINAL_GOLDEN_BYTES,
           (unsigned long)NPU_FC1_TILE_BASE,
           (unsigned long)NPU_FC1_TILE_BYTES,
           (unsigned long)NPU_FC2_TILE_BASE,
           (unsigned long)NPU_FC2_TILE_BYTES);
    return 0;
}

static int compare_layer_golden(const layer_desc_t *layer, u32 output_addr)
{
    const int16_t *actual = (const int16_t *)(UINTPTR)output_addr;
    const int16_t *golden =
        (const int16_t *)(UINTPTR)g_layer_golden_addr[layer->id];
    u32 value_count = layer->output_bytes / 2U;
    u32 index;
    u32 mismatches = 0U;

    if ((layer->output_bytes & 1U) != 0U) {
        return -1;
    }
    Xil_DCacheInvalidateRange((INTPTR)output_addr, layer->output_bytes);
    Xil_DCacheInvalidateRange((INTPTR)g_layer_golden_addr[layer->id],
                              layer->output_bytes);
    for (index = 0U; index < value_count; ++index) {
        u32 golden_index = index;
        if (layer->kind == LAYER_CONV) {
            u32 time_index = index / layer->cout;
            u32 channel_index = index % layer->cout;
            /* DMA/chain tensor is [T][C]; handoff Golden is canonical [C][T]. */
            golden_index = channel_index * layer->tout + time_index;
        }
        if (actual[index] != golden[golden_index]) {
            if (mismatches < 16U) {
                printf("LAYER_GOLDEN_MISMATCH,id=%lu,name=%s,actual_index=%lu,golden_index=%lu,actual=%d,expected=%d\r\n",
                       (unsigned long)layer->id, layer->name,
                       (unsigned long)index, (unsigned long)golden_index,
                       (int)actual[index], (int)golden[golden_index]);
            }
            ++mismatches;
        }
    }

    if (mismatches != 0U) {
        printf("LAYER_GOLDEN_FAIL,id=%lu,name=%s,values=%lu,mismatches=%lu\r\n",
               (unsigned long)layer->id, layer->name,
               (unsigned long)value_count,
               (unsigned long)mismatches);
        return -1;
    }
    printf("LAYER_GOLDEN_PASS,id=%lu,name=%s,values=%lu,mismatches=0,output=0x%08lx\r\n",
           (unsigned long)layer->id, layer->name,
           (unsigned long)value_count,
           (unsigned long)output_addr);
    return 0;
}

static int run_conv2_standalone_diag(void)
{
    const layer_desc_t *layer = &g_layers[1];
    bench_sample_t sample;
    int rc;

    printf("\r\n=== CONV2 STANDALONE DIAG START ===\r\n");
    printf("Input: preserved exact Conv1 output at ACT_A=0x%08lx; "
           "output ACT_B=0x%08lx\r\n",
           (unsigned long)NPU_ACT_A_BASE,
           (unsigned long)NPU_ACT_B_BASE);
    printf("Policy: DMA reset + NPU soft reset before Conv2; no Conv1 rerun.\r\n");

    rc = prepare_real_model_payloads();
    if (rc != 0) {
        printf("CONV2_STANDALONE_DIAG: payload preparation failed rc=%d\r\n", rc);
        return -1;
    }
    /* ACT_A was produced by S2MM and subsequently read-only compared. */
    Xil_DCacheInvalidateRange((INTPTR)NPU_ACT_A_BASE, layer->input_bytes);
    dma_reset();
    rc = npu_unified_soft_reset();
    if (rc != 0) {
        printf("CONV2_STANDALONE_DIAG: soft reset failed rc=%d\r\n", rc);
        return -2;
    }
    rc = run_layer_once(layer, NPU_ACT_A_BASE, NPU_ACT_B_BASE, 0,
                        BENCH_CHAIN_TIMEOUT_SECONDS, &sample);
    if (rc != 0) {
        printf("CONV2_STANDALONE_DIAG: operation failed rc=%d status_layer=%lu "
               "TXSR=0x%08lx RXSR=0x%08lx\r\n",
               rc, (unsigned long)npu_unified_read_layer_id(),
               (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
               (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
        return -3;
    }
    rc = compare_layer_golden(layer, NPU_ACT_B_BASE);
    printf("CONV2_STANDALONE_DIAG: %s\r\n", (rc == 0) ? "PASS" : "FAIL");
    return rc;
}

static int run_no_reset_chain(int use_real_model, int compare_golden,
                              u32 *final_output_addr)
{
    const u32 layer_count = (u32)(sizeof(g_layers) / sizeof(g_layers[0]));
    u32 input_addr = NPU_INPUT_BASE;
    u32 output_addr = NPU_ACT_A_BASE;
    u32 layer_index;
    uint64_t traffic_bytes = 0ULL;
    XTime operation_sum = 0U;
    XTime chain_start;
    XTime chain_total;
    int rc;

    printf("\r\n=== %s 12-LAYER CHAIN START ===\r\n",
           compare_golden ? "MODEL GOLDEN NO-RESET" :
           (use_real_model ? "LIVE MODEL NO-RESET" : "NO-RESET"));
    printf("Policy: npu_unified_init reset only; no reset inside or between layers.\r\n");
    printf("Routing: INPUT -> ACT_A/B ping-pong; FC activation is packed from the preceding S2MM output.\r\n");

    dma_reset();
    chain_start = timer_now();

    for (layer_index = 0U; layer_index < layer_count; ++layer_index) {
        const layer_desc_t *layer = &g_layers[layer_index];
        bench_sample_t sample;

        if ((layer_index > 0U) &&
            (g_layers[layer_index - 1U].output_bytes != layer->input_bytes)) {
            printf("CHAIN_FAULT: %s->%s size discontinuity %lu!=%lu\r\n",
                   g_layers[layer_index - 1U].name, layer->name,
                   (unsigned long)g_layers[layer_index - 1U].output_bytes,
                   (unsigned long)layer->input_bytes);
            return -1;
        }
        if (layer->kind == LAYER_FC) {
            rc = prepare_fc_chain_stream(layer, input_addr, use_real_model);
            if (rc != 0) {
                printf("CHAIN_FAULT: %s FC stream pack failed rc=%d\r\n",
                       layer->name, rc);
                return -2;
            }
        }

        printf("CHAIN_BEGIN,%lu,%s,input=0x%08lx,output=0x%08lx\r\n",
               (unsigned long)layer->id, layer->name,
               (unsigned long)input_addr, (unsigned long)output_addr);
        rc = run_layer_once(layer, input_addr, output_addr, 0,
                            BENCH_CHAIN_TIMEOUT_SECONDS, &sample);
        if (rc != 0) {
            printf("CHAIN_FAULT,%lu,%s,rc=%d,status_layer=%lu,TXSR=0x%08lx,RXSR=0x%08lx\r\n",
                   (unsigned long)layer->id, layer->name, rc,
                   (unsigned long)npu_unified_read_layer_id(),
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
            return -3;
        }
        if (compare_golden != 0) {
            rc = compare_layer_golden(layer, output_addr);
            if (rc != 0) {
                printf("CHAIN_FAULT,%lu,%s,stage=golden_compare,rc=%d\r\n",
                       (unsigned long)layer->id, layer->name, rc);
                return -4;
            }
        }

        operation_sum += sample.total_ticks;
        traffic_bytes += (uint64_t)sample.traffic_bytes;
        printf("CHAIN_PASS,%lu,%s,total_us=", (unsigned long)layer->id,
               layer->name);
        print_time_us(sample.total_ticks);
        printf(",tx_done_us=");
        print_time_us(sample.tx_done_ticks);
        printf(",rx_done_us=");
        print_time_us(sample.rx_done_ticks);
        printf(",npu_done_us=");
        print_time_us(sample.npu_done_ticks);
        printf("\r\n");
        print_convn_profile(layer, &sample);

        /* Keep an independent real-time producer serviced between layers.
         * The longest live layer is shorter than the Audio Formatter ring. */
        if ((g_service_hook != NULL) && (g_service_hook() != 0)) {
            printf("CHAIN_FAULT,%lu,%s,stage=service_hook\r\n",
                   (unsigned long)layer->id, layer->name);
            return -5;
        }

        input_addr = output_addr;
        output_addr = (output_addr == NPU_ACT_A_BASE) ?
                      NPU_ACT_B_BASE : NPU_ACT_A_BASE;
    }

    chain_total = timer_now() - chain_start;
    {
        uint64_t ChainUs = ticks_to_ns(chain_total) / 1000ULL;
        g_last_run_us = (ChainUs > (uint64_t)UINT32_MAX) ?
                        UINT32_MAX : (u32)ChainUs;
    }
    printf("CHAIN_RESULT: PASS layers=%lu traffic_bytes=%llu operation_sum_us=",
           (unsigned long)layer_count, (unsigned long long)traffic_bytes);
    print_time_us(operation_sum);
    printf(" wall_us=");
    print_time_us(chain_total);
    printf("\r\n=== %s 12-LAYER CHAIN COMPLETE ===\r\n",
           compare_golden ? "MODEL GOLDEN NO-RESET" :
           (use_real_model ? "LIVE MODEL NO-RESET" : "NO-RESET"));
    if (compare_golden != 0) {
        printf("MODEL_GOLDEN_RESULT: PASS layers=12 final_values=%lu mismatches=0\r\n",
               (unsigned long)NPU_FINAL_GOLDEN_VALUES);
    }
    if (final_output_addr != NULL) {
        *final_output_addr = input_addr;
    }
    return 0;
}

/*
 * Diagnose operation-boundary re-entry one adjacent pair at a time.
 * Every pair starts from one explicit NPU/DMA reset. The source and target
 * then run back-to-back with no reset between them. A failed pair is reset
 * independently so it cannot hide later transition results.
 */
static int run_transition_sweep(void)
{
    const u32 layer_count = (u32)(sizeof(g_layers) / sizeof(g_layers[0]));
    u32 pair_index;
    u32 pass_count = 0U;
    u32 fail_count = 0U;

    printf("\r\n=== ADJACENT NO-RESET TRANSITION SWEEP START ===\r\n");
    printf("Policy: one reset before each pair; no reset between source and target.\r\n");

    for (pair_index = 0U; pair_index + 1U < layer_count; ++pair_index) {
        const layer_desc_t *source = &g_layers[pair_index];
        const layer_desc_t *target = &g_layers[pair_index + 1U];
        u32 source_input = (pair_index == 0U) ?
                           NPU_INPUT_BASE : NPU_ACT_A_BASE;
        u32 source_output = (pair_index == 0U) ?
                            NPU_ACT_A_BASE : NPU_ACT_B_BASE;
        u32 target_input = source_output;
        u32 target_output = (source_output == NPU_ACT_A_BASE) ?
                            NPU_ACT_B_BASE : NPU_ACT_A_BASE;
        bench_sample_t source_sample;
        bench_sample_t target_sample;
        int rc;

        printf("TRANSITION_BEGIN,%lu,%s->%s\r\n",
               (unsigned long)pair_index, source->name, target->name);

        dma_reset();
        rc = npu_unified_soft_reset();
        if (rc != 0) {
            printf("TRANSITION_FAIL,%lu,%s->%s,stage=pair_reset,rc=%d\r\n",
                   (unsigned long)pair_index, source->name, target->name, rc);
            ++fail_count;
            continue;
        }

        if (source->kind == LAYER_FC) {
            rc = prepare_fc_chain_stream(source, source_input, 0);
            if (rc != 0) {
                printf("TRANSITION_FAIL,%lu,%s->%s,stage=source_pack,rc=%d\r\n",
                       (unsigned long)pair_index, source->name, target->name, rc);
                ++fail_count;
                continue;
            }
        }
        rc = run_layer_once(source, source_input, source_output, 0,
                            BENCH_CHAIN_TIMEOUT_SECONDS, &source_sample);
        if (rc != 0) {
            printf("TRANSITION_FAIL,%lu,%s->%s,stage=source,rc=%d,status_layer=%lu,TXSR=0x%08lx,RXSR=0x%08lx\r\n",
                   (unsigned long)pair_index, source->name, target->name, rc,
                   (unsigned long)npu_unified_read_layer_id(),
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
            ++fail_count;
            dma_reset();
            (void)npu_unified_soft_reset();
            continue;
        }

        if (source->output_bytes != target->input_bytes) {
            printf("TRANSITION_FAIL,%lu,%s->%s,stage=size,%lu!=%lu\r\n",
                   (unsigned long)pair_index, source->name, target->name,
                   (unsigned long)source->output_bytes,
                   (unsigned long)target->input_bytes);
            ++fail_count;
            continue;
        }
        if (target->kind == LAYER_FC) {
            rc = prepare_fc_chain_stream(target, target_input, 0);
            if (rc != 0) {
                printf("TRANSITION_FAIL,%lu,%s->%s,stage=target_pack,rc=%d\r\n",
                       (unsigned long)pair_index, source->name, target->name, rc);
                ++fail_count;
                continue;
            }
        }

        rc = run_layer_once(target, target_input, target_output, 0,
                            BENCH_CHAIN_TIMEOUT_SECONDS, &target_sample);
        if (rc != 0) {
            printf("TRANSITION_FAIL,%lu,%s->%s,stage=target,rc=%d,status_layer=%lu,TXSR=0x%08lx,RXSR=0x%08lx\r\n",
                   (unsigned long)pair_index, source->name, target->name, rc,
                   (unsigned long)npu_unified_read_layer_id(),
                   (unsigned long)dma_status(XAXIDMA_DMA_TO_DEVICE),
                   (unsigned long)dma_status(XAXIDMA_DEVICE_TO_DMA));
            ++fail_count;
            dma_reset();
            (void)npu_unified_soft_reset();
            continue;
        }

        ++pass_count;
        printf("TRANSITION_PASS,%lu,%s->%s,source_us=",
               (unsigned long)pair_index, source->name, target->name);
        print_time_us(source_sample.total_ticks);
        printf(",target_us=");
        print_time_us(target_sample.total_ticks);
        printf("\r\n");
    }

    printf("TRANSITION_RESULT,pairs=%lu,pass=%lu,fail=%lu\r\n",
           (unsigned long)(layer_count - 1U),
           (unsigned long)pass_count, (unsigned long)fail_count);
    printf("=== ADJACENT NO-RESET TRANSITION SWEEP COMPLETE ===\r\n");
    return (fail_count == 0U) ? 0 : 1;
}

int npu_validation_reference_main(void)
{
    u32 layer_index;
    u32 run;
    u32 final_output_addr = 0U;
    int rc;
    XTime timer_self_test_delta;

    init_platform();
    rc = init_benchmark_timer(&timer_self_test_delta);
    printf("\r\n=== Unified NPU hardware validation application ===\r\n");
    printf("Scope: %s\r\n",
           BENCH_RUN_MODEL_GOLDEN_CHAIN ?
           "canonical 12-layer bit-exact Golden chain" :
           "protocol-valid synthetic timing/transition tests");
    printf("Timer=%lu Hz, assumed PL clock=%lu Hz, warmup=%lu, measured=%lu\r\n",
           (unsigned long)benchmark_timer_hz(),
           (unsigned long)BENCH_PL_CLOCK_HZ,
           (unsigned long)BENCH_WARMUP_RUNS,
           (unsigned long)BENCH_MEASURED_RUNS);
    printf("Per-sample soft reset=%s (excluded from timing)\r\n",
           BENCH_SOFT_RESET_EACH_RUN ? "ON" : "OFF");
    printf("Transition sweep=%s, no-reset chain=%s, cold benchmark=%s\r\n",
           BENCH_RUN_TRANSITION_SWEEP ? "ON" : "OFF",
           BENCH_RUN_NO_RESET_CHAIN ? "ON" : "OFF",
           BENCH_RUN_COLD_BENCHMARK ? "ON" : "OFF");
    if (rc != 0) {
        printf("FAULT: PS global timer did not advance (rc=%d); benchmark aborted\r\n",
               rc);
        cleanup_platform();
        return -1;
    }
    printf("Timer self-test delta=%llu ticks (PASS)\r\n",
           (unsigned long long)timer_self_test_delta);

        rc = npu_unified_init(NPU_REG_BASE);
    if (rc != 0) {
        printf("FAULT: unified NPU init failed rc=%d version=0x%08lx\r\n",
               rc, (unsigned long)npu_unified_read_version());
        cleanup_platform();
        return -1;
    }
    printf("NPU VERSION=0x%08lx, DMA=0x%08lx\r\n",
           (unsigned long)npu_unified_read_version(),
           (unsigned long)XPAR_AXI_DMA_0_BASEADDR);

    rc = init_dma();
    if (rc != 0) {
        printf("FAULT: DMA init failed rc=%d\r\n", rc);
        cleanup_platform();
        return -1;
    }

    if (BENCH_RUN_CONV2_STANDALONE_DIAG != 0U) {
        rc = run_conv2_standalone_diag();
        printf("\r\n=== APPLICATION COMPLETE ===\r\n");
        cleanup_platform();
        return (rc == 0) ? 0 : -1;
    }

    if (BENCH_RUN_MODEL_GOLDEN_CHAIN != 0U) {
        printf("Mode: canonical model/input, final FC2 bit-exact Golden gate\r\n");
        rc = prepare_real_model_payloads();
        if (rc == 0) {
            rc = run_no_reset_chain(1, 1, &final_output_addr);
        }
        if (rc != 0) {
            printf("MODEL_GOLDEN_GATE: FAIL rc=%d\r\n", rc);
            dma_reset();
            (void)npu_unified_soft_reset();
            cleanup_platform();
            return -1;
        }
        printf("MODEL_GOLDEN_GATE: PASS final_output=0x%08lx\r\n\r\n=== APPLICATION COMPLETE ===\r\n",
               (unsigned long)final_output_addr);
        cleanup_platform();
        return 0;
    }

    prepare_zero_payloads();

    if (BENCH_RUN_TRANSITION_SWEEP != 0U) {
        (void)run_transition_sweep();
    }

    if (BENCH_RUN_NO_RESET_CHAIN != 0U) {
        rc = run_no_reset_chain(0, 0, NULL);
        if (rc != 0) {
            dma_reset();
            (void)npu_unified_soft_reset();
            cleanup_platform();
            return -1;
        }
    }

    if (BENCH_RUN_COLD_BENCHMARK == 0U) {
        printf("\r\n=== APPLICATION COMPLETE ===\r\n");
        cleanup_platform();
        return 0;
    }
    if ((BENCH_RUN_TRANSITION_SWEEP != 0U) ||
        (BENCH_RUN_NO_RESET_CHAIN != 0U)) {
        prepare_zero_payloads();
    }

    printf("CSV_HEADER,id,name,kind,tx_bytes,rx_bytes,min_us,avg_us,max_us,setup_avg_us,load_avg_us,est_pl_cycles,io_MBps\r\n");

    for (layer_index = 0U; layer_index <
         (u32)(sizeof(g_layers) / sizeof(g_layers[0])); ++layer_index) {
        const layer_desc_t *layer = &g_layers[layer_index];
        bench_stats_t stats;

        if ((BENCH_LAYER_MASK & (1U << layer->id)) == 0U) {
            continue;
        }
        stats_init(&stats, layer_tx_bytes(layer) + layer->output_bytes);
        printf("\r\n-- Layer %lu %s: TX=%lu B RX=%lu B --\r\n",
               (unsigned long)layer->id, layer->name,
               (unsigned long)layer_tx_bytes(layer),
               (unsigned long)layer->output_bytes);

        for (run = 0U; run < (BENCH_WARMUP_RUNS + BENCH_MEASURED_RUNS); ++run) {
            bench_sample_t sample;
            int warmup = run < BENCH_WARMUP_RUNS;

            printf("BEGIN: %s %s%lu\r\n", layer->name,
                   warmup ? "warmup" : "run",
                   (unsigned long)(warmup ? run : (run - BENCH_WARMUP_RUNS)));
            if (layer->kind == LAYER_FC) {
                rc = prepare_fc_chain_stream(layer, NPU_ACT_A_BASE, 0);
                if (rc != 0) {
                    printf("FAULT: %s synthetic FC stream pack rc=%d\r\n",
                           layer->name, rc);
                    cleanup_platform();
                    return -1;
                }
            }
            rc = run_layer_once(layer, NPU_ACT_A_BASE, NPU_ACT_B_BASE,
                                BENCH_SOFT_RESET_EACH_RUN != 0U,
                                BENCH_TIMEOUT_SECONDS, &sample);
            if (rc != 0) {
                printf("FAULT: %s iteration %lu failed rc=%d\r\n",
                       layer->name, (unsigned long)run, rc);
                dma_reset();
                (void)npu_unified_soft_reset();
                cleanup_platform();
                return -1;
            }
            print_sample(layer, warmup ? run : (run - BENCH_WARMUP_RUNS),
                         &sample, warmup);
            if (!warmup) {
                stats_add(&stats, &sample);
            }
        }
        print_summary(layer, &stats);
    }

    printf("\r\n=== BENCHMARK COMPLETE ===\r\n");
    cleanup_platform();
    return 0;
}

int NpuInference_Init(void)
{
    XTime TimerDelta;
    int Rc;

    Rc = init_benchmark_timer(&TimerDelta);
    if (Rc != 0) {
        printf("NPU_INIT_FAIL,timer_rc=%d\r\n", Rc);
        return Rc;
    }
    Rc = npu_unified_init(NPU_REG_BASE);
    if (Rc != 0) {
        printf("NPU_INIT_FAIL,npu_rc=%d,version=0x%08lx\r\n", Rc,
               (unsigned long)npu_unified_read_version());
        return Rc;
    }
    Rc = init_dma();
    if (Rc != 0) {
        printf("NPU_INIT_FAIL,dma_rc=%d\r\n", Rc);
        return Rc;
    }
    printf("NPU_READY,version=0x%08lx,npu=0x%08lx,dma=0x%08lx\r\n",
           (unsigned long)npu_unified_read_version(),
           (unsigned long)NPU_REG_BASE,
           (unsigned long)XPAR_AXI_DMA_0_BASEADDR);
    return 0;
}

int NpuInference_PrepareModel(void)
{
    return prepare_real_model_payloads();
}

int NpuInference_Run(u32 *FinalOutputAddress)
{
    int Rc;

    g_last_run_us = 0U;
    Rc = npu_unified_soft_reset();
    if (Rc != 0) {
        printf("NPU_RUN_FAIL,soft_reset_rc=%d\r\n", Rc);
        return Rc;
    }
    /* Real model and FC tiles, but no Golden comparison for live audio. */
    Rc = run_no_reset_chain(1, 0, FinalOutputAddress);
    if (Rc != 0) {
        printf("NPU_RUN_FAIL,chain_rc=%d\r\n", Rc);
    }
    return Rc;
}

void NpuInference_SetServiceHook(NpuInferenceServiceHook Hook)
{
    g_service_hook = Hook;
}

u32 NpuInference_GetLastRunUs(void)
{
    return g_last_run_us;
}

typedef struct {
    u32 ClassIndex;
    int16_t LogitQ10;
} npu_demo_result_t;

/* Sigmoid(x) in permille for x=0.00..8.00 at 0.25 intervals. */
static const u16 g_sigmoid_permille_lut[] = {
    500U, 562U, 622U, 679U, 731U, 777U, 818U, 852U, 881U,
    905U, 924U, 940U, 953U, 963U, 971U, 977U, 982U, 986U,
    989U, 991U, 993U, 995U, 996U, 997U, 998U, 998U, 998U,
    999U, 999U, 999U, 999U, 1000U, 1000U
};

static u32 NpuInference_LogitToPermille(int16_t LogitQ10)
{
    int32_t SignedValue = (int32_t)LogitQ10;
    u32 Magnitude = (SignedValue < 0) ?
        (u32)(-SignedValue) : (u32)SignedValue;
    u32 PositiveScore;

    if (Magnitude >= 8192U) {
        PositiveScore = 1000U;
    } else {
        u32 Segment = Magnitude >> 8;
        u32 Fraction = Magnitude & 0xFFU;
        u32 Lower = g_sigmoid_permille_lut[Segment];
        u32 Upper = g_sigmoid_permille_lut[Segment + 1U];

        PositiveScore = Lower +
            (((Upper - Lower) * Fraction + 128U) >> 8);
    }
    return (SignedValue < 0) ? (1000U - PositiveScore) : PositiveScore;
}

static void NpuInference_SelectTopK(const int16_t *Result,
                                    npu_demo_result_t *TopResults)
{
    u32 Index;
    u32 Rank;

    for (Rank = 0U; Rank < NPU_DEMO_TOP_K; ++Rank) {
        TopResults[Rank].ClassIndex = 0U;
        TopResults[Rank].LogitQ10 = INT16_MIN;
    }
    for (Index = 0U; Index < NPU_INFERENCE_RESULT_VALUES; ++Index) {
        for (Rank = 0U; Rank < NPU_DEMO_TOP_K; ++Rank) {
            if (Result[Index] > TopResults[Rank].LogitQ10) {
                u32 Move;

                for (Move = NPU_DEMO_TOP_K - 1U; Move > Rank; --Move) {
                    TopResults[Move] = TopResults[Move - 1U];
                }
                TopResults[Rank].ClassIndex = Index;
                TopResults[Rank].LogitQ10 = Result[Index];
                break;
            }
        }
    }
}

static void NpuInference_PrintRawResults(const int16_t *Result,
                                         u32 FinalOutputAddress)
{
    u32 Index;

    printf("NPU_RESULTS_BEGIN,count=%lu,address=0x%08lx\r\n",
           (unsigned long)NPU_INFERENCE_RESULT_VALUES,
           (unsigned long)FinalOutputAddress);
    for (Index = 0U; Index < NPU_INFERENCE_RESULT_VALUES; ++Index) {
        printf("NPU_RESULT,%lu,%d\r\n",
               (unsigned long)Index, (int)Result[Index]);
    }
    printf("NPU_RESULTS_END,count=%lu\r\n",
           (unsigned long)NPU_INFERENCE_RESULT_VALUES);
}

void NpuInference_PrintDemoResults(u32 FinalOutputAddress,
                                   const char *InputModeName)
{
    const int16_t *Result = (const int16_t *)(UINTPTR)FinalOutputAddress;
    npu_demo_result_t TopResults[NPU_DEMO_TOP_K];
    u32 Rank;

    Xil_DCacheInvalidateRange((INTPTR)FinalOutputAddress,
                              NPU_FINAL_GOLDEN_BYTES);
    NpuInference_SelectTopK(Result, TopResults);

    printf("\r\n============================================================\r\n");
    printf("                 NPU AUDIO EVENT RESULT\r\n");
    printf("============================================================\r\n");
    printf(" Input source : %s\r\n", InputModeName);
    printf(" Model output : 527 AudioSet classes (independent scores)\r\n");
    printf("------------------------------------------------------------\r\n");
    for (Rank = 0U; Rank < NPU_DEMO_TOP_K; ++Rank) {
        u32 ClassIndex = TopResults[Rank].ClassIndex;
        u32 Score = NpuInference_LogitToPermille(
            TopResults[Rank].LogitQ10);

        printf(" %lu. %-40s %3lu.%lu%%\r\n",
               (unsigned long)(Rank + 1U),
               g_npu_class_labels[ClassIndex],
               (unsigned long)(Score / 10U),
               (unsigned long)(Score % 10U));
    }
    printf("------------------------------------------------------------\r\n");
    {
        u32 TopScore = NpuInference_LogitToPermille(
            TopResults[0].LogitQ10);
        printf(" Top event    : %s (%lu.%lu%%)\r\n",
               g_npu_class_labels[TopResults[0].ClassIndex],
               (unsigned long)(TopScore / 10U),
               (unsigned long)(TopScore % 10U));
    }
    printf("============================================================\r\n");

    if (NPU_UART_PRINT_RAW_RESULTS != 0U) {
        NpuInference_PrintRawResults(Result, FinalOutputAddress);
    }
}
