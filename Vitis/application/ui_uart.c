#include "ui_uart.h"

#include "xil_printf.h"

void UiUart_Init(void)
{
    /* PS UART1 is initialized as 115200 8-N-1 by ps7_init/BSP. */
}

void UiUart_SendResult(const ui_result_t *Result)
{
    if (Result == NULL) {
        return;
    }

    xil_printf(
        "RESULT,%u,%d,%d,%u,%u,%u,%u,%u,%u\r\n",
        (unsigned int)Result->seq,
        (int)Result->level,
        Result->winner_class,
        (unsigned int)Result->winner_probability_1e4,
        (unsigned int)Result->activity_1e4,
        (unsigned int)Result->intrusion_damage_1e4,
        (unsigned int)Result->critical_1e4,
        (unsigned int)Result->facility_emergency_1e4,
        (unsigned int)Result->inference_us);
}
