import argparse
import os
import re
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    input_csv = os.path.join(args.input, 'new_data.csv')
    df = pd.read_csv(input_csv)

    # Basic type coercions for numeric computations
    numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    os.makedirs(args.output, exist_ok=True)

    # ASSERTION_START
    def _check_sleep_disorder_values(s):
        m = s.notna()
        if m.any():
            return s[m].isin(['Insomnia', 'Sleep Apnea']).all()
        return True
    assert _check_sleep_disorder_values(df['Sleep Disorder'])
    # ASSERTION_END

    has_known_disorder = df['Sleep Disorder'].notna()

    # ASSERTION_START
    def _check_bp(series):
        m = series.notna()
        pattern = r'^\d{2,3}/\d{2}$'
        if not series[m].str.match(pattern).all():
            return False
        parts = series[m].str.split('/', n=1, expand=True)
        sys = pd.to_numeric(parts[0], errors='coerce')
        dia = pd.to_numeric(parts[1], errors='coerce')
        rng = sys.between(90, 200, inclusive='both')
        cmp_ = sys > dia
        return rng.all() and cmp_.all()
    assert _check_bp(df['Blood Pressure'])
    # ASSERTION_END

    bp_parts = df['Blood Pressure'].str.split('/', n=1, expand=True)
    df['Systolic'] = pd.to_numeric(bp_parts[0], errors='coerce')
    df['Diastolic'] = pd.to_numeric(bp_parts[1], errors='coerce')

    def bp_risk_row(sys, dia):
        # Simple clinical staging proxy
        if (sys >= 180) or (dia >= 120):
            return 4
        if (sys >= 140) or (dia >= 90):
            return 3
        if (130 <= sys <= 139) or (80 <= dia <= 89):
            return 2
        if (120 <= sys <= 129) and (dia < 80):
            return 1
        return 0

    df['bp_risk'] = [bp_risk_row(s, d) for s, d in zip(df['Systolic'], df['Diastolic'])]

    # ASSERTION_START
    def _check_steps_pal(d):
        steps = pd.to_numeric(d['Daily Steps'], errors='coerce')
        pal = pd.to_numeric(d['Physical Activity Level'], errors='coerce')
        mask = steps.notna() & pal.notna()
        if mask.sum() < 2:
            return False
        corr = steps[mask].corr(pal[mask], method='pearson')
        per_record_ok = (steps[mask] >= pal[mask] * 30).all()
        return (corr is not None) and (corr > 0.6) and per_record_ok
    assert _check_steps_pal(df)
    # ASSERTION_END

    df['OverageSteps'] = df['Daily Steps'] - df['Physical Activity Level'] * 30
    df['ActivityBonus'] = np.sqrt(df['OverageSteps'])

    # ASSERTION_START
    def _check_duration_qos_corr(d):
        dur = pd.to_numeric(d['Sleep Duration'], errors='coerce')
        qos = pd.to_numeric(d['Quality of Sleep'], errors='coerce')
        mask = dur.notna() & qos.notna()
        if mask.sum() < 2:
            return False
        corr = dur[mask].corr(qos[mask], method='pearson')
        return (corr is not None) and (corr > 0.3)
    assert _check_duration_qos_corr(df)
    # ASSERTION_END

    m, b = np.polyfit(df['Sleep Duration'], df['Quality of Sleep'], 1)
    df['PredictedQoS'] = (m * df['Sleep Duration'] + b).clip(lower=df['Quality of Sleep'].min(), upper=df['Quality of Sleep'].max())

    # ASSERTION_START
    def _check_stress_qos_anticorr(d):
        stress = pd.to_numeric(d['Stress Level'], errors='coerce')
        qos = pd.to_numeric(d['Quality of Sleep'], errors='coerce')
        mask = stress.notna() & qos.notna()
        if mask.sum() < 2:
            return False
        corr = stress[mask].corr(qos[mask], method='spearman')
        return (corr is not None) and (corr < -0.3)
    assert _check_stress_qos_anticorr(df)
    # ASSERTION_END

    df['SleepDebt'] = np.where(df['Sleep Duration'] < 7.0, 7.0 - df['Sleep Duration'], np.where(df['Sleep Duration'] > 9.0, df['Sleep Duration'] - 9.0, 0.0))
    df['QoSGap'] = np.maximum(0.0, df['PredictedQoS'] - df['Quality of Sleep'])
    df['StressPenalty'] = np.maximum(0.0, df['Stress Level'] - 5) * 0.5

    df['risk_score'] = (
        2.2 * df['bp_risk'] +
        1.0 * df['SleepDebt'] +
        np.maximum(0.0, 6 - df['Quality of Sleep']) +
        0.6 * np.maximum(0.0, df['Stress Level'] - 5) -
        0.15 * df['ActivityBonus'] +
        0.7 * df['QoSGap']
    )

    df.loc[has_known_disorder, 'risk_score'] += 8.0

    def tier_row(row):
        if pd.notna(row['Sleep Disorder']):
            return 'Immediate'
        if row['risk_score'] >= 7.5:
            return 'High'
        if row['risk_score'] >= 4.0:
            return 'Medium'
        return 'Low'

    df['priority_tier'] = df.apply(tier_row, axis=1)

    def build_recommendations(row):
        recs = []
        if pd.notna(row['Sleep Disorder']):
            recs.append('Clinical follow-up for diagnosed condition')
        if row['Sleep Duration'] < 7.0:
            recs.append('Extend nightly time-in-bed by 30-60 minutes')
        if row['Sleep Duration'] > 9.0:
            recs.append('Align schedule to consistent 7-9h window')
        if row['Quality of Sleep'] < 6:
            recs.append('Sleep hygiene coaching and stimulus control')
        if row['Stress Level'] >= 7:
            recs.append('Daily stress-reduction routine (breathing, CBT-i)')
        if row['bp_risk'] >= 3:
            recs.append('Home BP monitoring and PCP evaluation')
        if row['ActivityBonus'] < 5:
            recs.append('Progressive step goal increase')
        return '; '.join(recs) if recs else 'Maintain current routine'

    df['recommendations'] = df.apply(build_recommendations, axis=1)

    priority_rank = {'Immediate': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    df['priority_rank'] = df['priority_tier'].map(priority_rank)

    prioritized_cols = [
        'Person ID', 'Gender', 'Age', 'Sleep Disorder', 'Systolic', 'Diastolic', 'bp_risk',
        'Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Daily Steps', 'Physical Activity Level',
        'risk_score', 'priority_tier'
    ]

    prioritized = df[prioritized_cols].sort_values(by=['priority_rank', 'risk_score'], ascending=[True, False])
    prioritized.to_csv(os.path.join(args.output, 'prioritized_members.csv'), index=False)

    recs_out = df[['Person ID', 'priority_tier', 'risk_score', 'recommendations']].sort_values(by=['priority_rank', 'risk_score'], ascending=[True, False])
    recs_out.to_csv(os.path.join(args.output, 'coaching_recommendations.csv'), index=False)


if __name__ == '__main__':
    main()
