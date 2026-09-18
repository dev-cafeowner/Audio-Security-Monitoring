#include "npu_unified_driver.h"

#include <stddef.h>
#include "xil_io.h"

#define NPU_REG_CTRL      0x00U
#define NPU_REG_STATUS    0x04U
#define NPU_REG_MODEL     0x08U
#define NPU_REG_INPUT     0x0CU
#define NPU_REG_OUTPUT    0x10U
#define NPU_REG_VERSION   0x14U
#define NPU_REG_LAYER_ID  0x1CU

#define NPU_CTRL_START       (1U << 0)
#define NPU_CTRL_SOFT_RESET  (1U << 1)
#define NPU_STATUS_BUSY      (1U << 0)
#define NPU_STATUS_DONE      (1U << 1)
#define NPU_STATUS_ERROR     (1U << 2)
#define NPU_STATUS_W1C       (NPU_STATUS_DONE | NPU_STATUS_ERROR)

static UINTPTR g_npu_base;

static u32 reg_read(u32 offset)
{
    return Xil_In32(g_npu_base + offset);
}

static void reg_write(u32 offset, u32 value)
{
    Xil_Out32(g_npu_base + offset, value);
}

u32 npu_unified_read_version(void)
{
    return reg_read(NPU_REG_VERSION);
}

u32 npu_unified_read_layer_id(void)
{
    return reg_read(NPU_REG_LAYER_ID) & 0xFU;
}

void npu_unified_read_status(npu_unified_status_t *status)
{
    u32 raw = reg_read(NPU_REG_STATUS);

    if (status == NULL) {
        return;
    }
    status->raw = raw;
    status->busy = (u8)((raw & NPU_STATUS_BUSY) != 0U);
    status->done = (u8)((raw & NPU_STATUS_DONE) != 0U);
    status->error = (u8)((raw & NPU_STATUS_ERROR) != 0U);
    status->error_code = (u8)((raw >> 8) & 0xFFU);
}

void npu_unified_clear_status(void)
{
    reg_write(NPU_REG_STATUS, NPU_STATUS_W1C);
}

int npu_unified_soft_reset(void)
{
    npu_unified_status_t status;
    u32 poll;

    reg_write(NPU_REG_CTRL, NPU_CTRL_SOFT_RESET);
    for (poll = 0U; poll < 1000000U; ++poll) {
        npu_unified_read_status(&status);
        if (!status.busy) {
            npu_unified_clear_status();
            return 0;
        }
    }
    return -1;
}

int npu_unified_init(UINTPTR base_addr)
{
    g_npu_base = base_addr;
    if (npu_unified_read_version() != NPU_UNIFIED_VERSION_MAGIC) {
        return -1;
    }
    return npu_unified_soft_reset();
}

int npu_unified_configure(u8 layer_id, u32 model_addr,
                          u32 input_addr, u32 output_addr)
{
    npu_unified_status_t status;

    if ((layer_id >= 12U) ||
        (((model_addr | input_addr | output_addr) & 0x3U) != 0U)) {
        return -1;
    }

    npu_unified_read_status(&status);
    if (status.busy) {
        return -2;
    }

    npu_unified_clear_status();
    reg_write(NPU_REG_LAYER_ID, (u32)layer_id);
    reg_write(NPU_REG_MODEL, model_addr);
    reg_write(NPU_REG_INPUT, input_addr);
    reg_write(NPU_REG_OUTPUT, output_addr);

    if ((npu_unified_read_layer_id() != (u32)layer_id) ||
        (reg_read(NPU_REG_MODEL) != model_addr) ||
        (reg_read(NPU_REG_INPUT) != input_addr) ||
        (reg_read(NPU_REG_OUTPUT) != output_addr)) {
        return -3;
    }
    return 0;
}

int npu_unified_start(void)
{
    npu_unified_status_t status;

    npu_unified_read_status(&status);
    if (status.busy) {
        return -1;
    }
    reg_write(NPU_REG_CTRL, NPU_CTRL_START);
    npu_unified_read_status(&status);
    if (status.error) {
        return -(int)(0x100U | status.error_code);
    }
    return status.busy ? 0 : -2;
}
