#ifndef NPU_UNIFIED_DRIVER_H
#define NPU_UNIFIED_DRIVER_H

#include "xil_types.h"

#define NPU_UNIFIED_VERSION_MAGIC 0x4E505531U

typedef struct {
    u8 busy;
    u8 done;
    u8 error;
    u8 error_code;
    u32 raw;
} npu_unified_status_t;

int npu_unified_init(UINTPTR base_addr);
int npu_unified_soft_reset(void);
int npu_unified_configure(u8 layer_id, u32 model_addr,
                          u32 input_addr, u32 output_addr);
int npu_unified_start(void);
void npu_unified_read_status(npu_unified_status_t *status);
void npu_unified_clear_status(void);
u32 npu_unified_read_layer_id(void);
u32 npu_unified_read_version(void);

#endif
