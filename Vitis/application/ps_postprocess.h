#ifndef PS_POSTPROCESS_H
#define PS_POSTPROCESS_H

#include "ui_types.h"

#define AUDIOSET_CLASS_COUNT 527U

/*
 * Demo-calibrated policy. The selected AudioSet clips leave at least roughly
 * 8 percentage points of FP32 margin above these thresholds. Re-tune with
 * real site recordings before treating them as deployment thresholds.
 */
#define PS_POSTPROCESS_ACTIVITY_THRESHOLD_1E4   4500U
#define PS_POSTPROCESS_INTRUSION_THRESHOLD_1E4  2000U
#define PS_POSTPROCESS_CRITICAL_THRESHOLD_1E4   4000U
#define PS_POSTPROCESS_EMERGENCY_THRESHOLD_1E4  3500U

/*
 * Convert 527 signed INT16 Q10 logits into one UI result.
 * seq and inference_us are runtime metadata and are filled by the caller.
 */
int PsPostprocess_Run(const s16 Logits[AUDIOSET_CLASS_COUNT],
                      ui_result_t *Result);

#endif /* PS_POSTPROCESS_H */
