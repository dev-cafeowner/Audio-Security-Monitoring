#include "ps_postprocess.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    u16 Score1e4;
    u16 ClassIndex;
} group_winner_t;

/* AudioSet class indices used by the PC security-monitoring UI. */
static const u16 g_activity_classes[] = {
    8U, 11U, 14U, 51U, 52U, 53U,
    354U, 355U, 357U, 358U, 359U
};

static const u16 g_intrusion_classes[] = {
    440U, 441U, 443U, 460U, 461U, 466U,
    469U, 470U, 478U, 480U
};

static const u16 g_critical_classes[] = {
    426U, 427U, 428U, 429U, 430U, 436U
};

static const u16 g_emergency_classes[] = {
    298U, 388U, 395U, 396U, 397U, 399U, 400U
};

/* Sigmoid(x) in permille for x=0.00..8.00 at 0.25 intervals. */
static const u16 g_sigmoid_permille_lut[] = {
    500U, 562U, 622U, 679U, 731U, 777U, 818U, 852U, 881U,
    905U, 924U, 940U, 953U, 963U, 971U, 977U, 982U, 986U,
    989U, 991U, 993U, 995U, 996U, 997U, 998U, 998U, 998U,
    999U, 999U, 999U, 999U, 1000U, 1000U
};

static u16 LogitToScore1e4(s16 LogitQ10)
{
    int32_t SignedValue = (int32_t)LogitQ10;
    u32 Magnitude = (SignedValue < 0) ?
        (u32)(-SignedValue) : (u32)SignedValue;
    u32 PositivePermille;

    if (Magnitude >= 8192U) {
        PositivePermille = 1000U;
    } else {
        u32 Segment = Magnitude >> 8;
        u32 Fraction = Magnitude & 0xFFU;
        u32 Lower = g_sigmoid_permille_lut[Segment];
        u32 Upper = g_sigmoid_permille_lut[Segment + 1U];

        PositivePermille = Lower +
            (((Upper - Lower) * Fraction + 128U) >> 8);
    }
    if (SignedValue < 0) {
        PositivePermille = 1000U - PositivePermille;
    }
    return (u16)(PositivePermille * 10U);
}

static group_winner_t SelectGroupWinner(const s16 *Logits,
                                         const u16 *Classes,
                                         size_t ClassCount)
{
    group_winner_t Winner = {0U, 0U};
    size_t Index;

    for (Index = 0U; Index < ClassCount; ++Index) {
        u16 ClassIndex = Classes[Index];
        u16 Score = LogitToScore1e4(Logits[ClassIndex]);

        if ((Index == 0U) || (Score > Winner.Score1e4)) {
            Winner.Score1e4 = Score;
            Winner.ClassIndex = ClassIndex;
        }
    }
    return Winner;
}

int PsPostprocess_Run(const s16 Logits[AUDIOSET_CLASS_COUNT],
                      ui_result_t *Result)
{
    group_winner_t Activity;
    group_winner_t Intrusion;
    group_winner_t Critical;
    group_winner_t Emergency;
    u32 GlobalWinner = 0U;
    u32 Index;

    if ((Logits == NULL) || (Result == NULL)) {
        return -1;
    }

    memset(Result, 0, sizeof(*Result));
    for (Index = 1U; Index < AUDIOSET_CLASS_COUNT; ++Index) {
        if (Logits[Index] > Logits[GlobalWinner]) {
            GlobalWinner = Index;
        }
    }

    Activity = SelectGroupWinner(Logits, g_activity_classes,
        sizeof(g_activity_classes) / sizeof(g_activity_classes[0]));
    Intrusion = SelectGroupWinner(Logits, g_intrusion_classes,
        sizeof(g_intrusion_classes) / sizeof(g_intrusion_classes[0]));
    Critical = SelectGroupWinner(Logits, g_critical_classes,
        sizeof(g_critical_classes) / sizeof(g_critical_classes[0]));
    Emergency = SelectGroupWinner(Logits, g_emergency_classes,
        sizeof(g_emergency_classes) / sizeof(g_emergency_classes[0]));

    Result->activity_1e4 = Activity.Score1e4;
    Result->intrusion_damage_1e4 = Intrusion.Score1e4;
    Result->critical_1e4 = Critical.Score1e4;
    Result->facility_emergency_1e4 = Emergency.Score1e4;
    Result->winner_class = (int)GlobalWinner;
    Result->winner_probability_1e4 =
        LogitToScore1e4(Logits[GlobalWinner]);
    Result->level = SEC_NORMAL;

    /* Higher severity wins when more than one group crosses its threshold. */
    if (Emergency.Score1e4 >= PS_POSTPROCESS_EMERGENCY_THRESHOLD_1E4) {
        Result->level = SEC_FACILITY_EMERGENCY;
        Result->winner_class = (int)Emergency.ClassIndex;
        Result->winner_probability_1e4 = Emergency.Score1e4;
    } else if (Critical.Score1e4 >= PS_POSTPROCESS_CRITICAL_THRESHOLD_1E4) {
        Result->level = SEC_CRITICAL;
        Result->winner_class = (int)Critical.ClassIndex;
        Result->winner_probability_1e4 = Critical.Score1e4;
    } else if (Intrusion.Score1e4 >= PS_POSTPROCESS_INTRUSION_THRESHOLD_1E4) {
        Result->level = SEC_INTRUSION_DAMAGE;
        Result->winner_class = (int)Intrusion.ClassIndex;
        Result->winner_probability_1e4 = Intrusion.Score1e4;
    } else if (Activity.Score1e4 >= PS_POSTPROCESS_ACTIVITY_THRESHOLD_1E4) {
        Result->level = SEC_ACTIVITY;
        Result->winner_class = (int)Activity.ClassIndex;
        Result->winner_probability_1e4 = Activity.Score1e4;
    }

    return 0;
}
