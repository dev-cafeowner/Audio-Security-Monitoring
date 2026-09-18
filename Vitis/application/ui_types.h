#ifndef UI_TYPES_H
#define UI_TYPES_H

#include "xil_types.h"

typedef enum
{
    SEC_NORMAL = 0,
    SEC_ACTIVITY = 1,
    SEC_INTRUSION_DAMAGE = 2,
    SEC_CRITICAL = 3,
    SEC_FACILITY_EMERGENCY = 4
} security_level_t;

/*
 * Probabilities and security group scores use a 1e4 scale:
 * 8624 means 0.8624, or 86.24 percent.
 */
typedef struct
{
    u32 seq;
    security_level_t level;
    int winner_class;
    u16 winner_probability_1e4;
    u16 activity_1e4;
    u16 intrusion_damage_1e4;
    u16 critical_1e4;
    u16 facility_emergency_1e4;
    u32 inference_us;
} ui_result_t;

#endif /* UI_TYPES_H */
