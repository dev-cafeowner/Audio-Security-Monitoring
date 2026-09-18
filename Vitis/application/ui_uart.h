#ifndef UI_UART_H
#define UI_UART_H

#include "ui_types.h"

void UiUart_Init(void);
void UiUart_SendResult(const ui_result_t *Result);

#endif /* UI_UART_H */
