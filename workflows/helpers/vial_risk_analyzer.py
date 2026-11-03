# -*- coding: utf-8 -*-
"""
Vial Risk Assessment - Identifies problematic vials before dispense.

No positioning changes. Just evaluation and reporting.
Risk increases as: volume drops + usage increases.
"""

import pandas as pd
from typing import Dict, List


class VialRiskAnalyzer:
    """Assess vial risk based on volume and usage patterns."""
    
    def __init__(self, logger=None):
        self.logger = logger or self._default_logger()
    
    @staticmethod
    def _default_logger():
        import logging
        return logging.getLogger(__name__)
    
    def analyze_vial_usage(
        self,
        vial_names: List[str],
        batch_df: pd.DataFrame,
        lash_e,
        vial_type: str = 'A'
    ) -> Dict[str, Dict]:
        """
        Extract per-vial usage facts from batch recipe.
        
        Returns:
            {vial_name: {
                'current_volume_ml': float,
                'dispense_count': int,
                'total_volume_required_ml': float,
                'remaining_after_all_ml': float,
                'utilization_pct': float,
            }}
        """
        analysis = {}
        vol_column = f'surf_{vial_type}_volume_ul'
        name_column = f'substock_{vial_type}_name'
        
        for vial_name in vial_names:
            try:
                # Get current volume
                current_vol_ml = lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume')
                current_vol_ml = float(current_vol_ml) if current_vol_ml else 0.0
                
                # Find all wells using this vial
                wells_using_vial = batch_df[batch_df[name_column] == vial_name]
                dispense_count = len(wells_using_vial)
                
                # Total volume required
                total_vol_ul = wells_using_vial[vol_column].sum() if dispense_count > 0 else 0.0
                total_vol_ml = total_vol_ul / 1000.0
                
                # Remaining after all dispenses
                remaining_ml = current_vol_ml - total_vol_ml
                
                # Utilization as percentage
                utilization_pct = (total_vol_ml / max(current_vol_ml, 0.01)) * 100
                
                analysis[vial_name] = {
                    'current_volume_ml': current_vol_ml,
                    'dispense_count': dispense_count,
                    'total_volume_required_ml': total_vol_ml,
                    'remaining_after_all_ml': remaining_ml,
                    'utilization_pct': utilization_pct,
                }
                
            except Exception as e:
                self.logger.warning(f"  Could not analyze {vial_name}: {e}")
                analysis[vial_name] = {
                    'current_volume_ml': 0.0,
                    'dispense_count': 0,
                    'total_volume_required_ml': 0.0,
                    'remaining_after_all_ml': 0.0,
                    'utilization_pct': 0.0,
                }
        
        return analysis
    
    def compute_risk_score(
        self,
        current_ml: float,
        remaining_ml: float,
        utilization_pct: float
    ) -> tuple:
        """
        Compute a numerical risk score from 0.0 (no risk) to 10.0 (maximum risk).

        Three components, each scaled 0-10:
          V (Volume)      - how low is the starting volume?   weight 35%
            0-10 mapped from [7 mL -> 0] to [0 mL -> 10]
          R (Remaining)   - how depleted will it be after?    weight 45%
            0-10 mapped from [5 mL -> 0] to [0 mL (or less) -> 10]
          U (Utilization) - what fraction of vial is consumed? weight 20%
            0-10 mapped from [0% -> 0] to [100% -> 10]

        Returns:
            (score, v_score, r_score, u_score)
        """
        # V: volume component (7 mL = 0 risk, 0 mL = 10 risk)
        v_score = max(0.0, min(10.0, (7.0 - current_ml) / 7.0 * 10.0))

        # R: remaining component (5 mL = 0 risk, <=0 mL = 10 risk)
        r_score = max(0.0, min(10.0, (5.0 - remaining_ml) / 5.0 * 10.0))

        # U: utilization component (0% = 0 risk, 100% = 10 risk)
        u_score = max(0.0, min(10.0, utilization_pct / 10.0))

        score = round(0.35 * v_score + 0.45 * r_score + 0.20 * u_score, 2)
        return score, round(v_score, 2), round(r_score, 2), round(u_score, 2)

    def compute_risk_level(
        self,
        current_ml: float,
        remaining_ml: float,
        utilization_pct: float
    ) -> str:
        """
        Categorize risk as SAFE | CAUTION | RISKY | CRITICAL.

        Based on actual operating thresholds:
        - 4-6 mL range is nominal
        - Lower volume + higher usage = increasing risk
        - Failures are rare but happen more as volume drops
        """
        # CRITICAL: highest risk
        if current_ml < 2.5 or remaining_ml < 0 or utilization_pct > 90:
            return "CRITICAL"

        # RISKY: significant risk
        if (2.5 <= current_ml < 4.0) or (0 <= remaining_ml < 0.5) or (75 <= utilization_pct <= 90):
            return "RISKY"

        # CAUTION: some risk
        if (4.0 <= current_ml < 5.5) or (0.5 <= remaining_ml < 1.0) or (50 <= utilization_pct < 75):
            return "CAUTION"

        # SAFE: low risk
        return "SAFE"
    
    def generate_risk_report(
        self,
        vial_names: List[str],
        analysis: Dict[str, Dict]
    ):
        """
        Log detailed risk assessment table.

        Score formula (0.0 = no risk, 10.0 = maximum risk):
          Score = 0.35*V + 0.45*R + 0.20*U
          V (Volume):      (7mL-current)/7  scaled 0-10   [low starting volume]
          R (Remaining):   (5mL-remaining)/5 scaled 0-10  [how depleted after dispense]
          U (Utilization): utilization%/10  scaled 0-10   [fraction of vial consumed]
        """
        COL = 150
        self.logger.info("\n" + "=" * COL)
        self.logger.info("VIAL RISK ASSESSMENT - Pre-Dispense Evaluation")
        self.logger.info("Score = 0.35*V + 0.45*R + 0.20*U  |  V=volume(0-10)  R=remaining(0-10)  U=utilization(0-10)  |  Total 0.0-10.0")
        self.logger.info("=" * COL)
        self.logger.info(
            f"{'Vial Name':<25} {'Current':>8}  {'Uses':>5}  {'Required':>9}  "
            f"{'Remaining':>9}  {'Util%':>6}  "
            f"{'V':>5}  {'R':>5}  {'U':>5}  {'Score':>6}  {'Level':<10}"
        )
        self.logger.info("-" * COL)

        risk_counts = {'SAFE': 0, 'CAUTION': 0, 'RISKY': 0, 'CRITICAL': 0}
        critical_vials = []

        for vial in vial_names:
            if vial in analysis:
                info = analysis[vial]
                risk_level = self.compute_risk_level(
                    info['current_volume_ml'],
                    info['remaining_after_all_ml'],
                    info['utilization_pct']
                )
                score, v_score, r_score, u_score = self.compute_risk_score(
                    info['current_volume_ml'],
                    info['remaining_after_all_ml'],
                    info['utilization_pct']
                )
                risk_counts[risk_level] += 1

                self.logger.info(
                    f"{vial:<25} {info['current_volume_ml']:>7.2f}mL "
                    f"{info['dispense_count']:>5} "
                    f"{info['total_volume_required_ml']:>8.2f}mL "
                    f"{info['remaining_after_all_ml']:>8.2f}mL "
                    f"{info['utilization_pct']:>5.1f}%  "
                    f"{v_score:>5.1f}  {r_score:>5.1f}  {u_score:>5.1f}  {score:>6.2f}  {risk_level:<10}"
                )

                if risk_level in ['RISKY', 'CRITICAL']:
                    critical_vials.append((vial, risk_level, score, info))

        self.logger.info("-" * COL)
        self.logger.info(
            f"Summary: {risk_counts['SAFE']} SAFE | {risk_counts['CAUTION']} CAUTION | "
            f"{risk_counts['RISKY']} RISKY | {risk_counts['CRITICAL']} CRITICAL"
        )

        if critical_vials:
            self.logger.warning("\n!! ATTENTION: Problem vials detected:")
            for vial, risk_level, score, info in critical_vials:
                self.logger.warning(f"  [{risk_level}] {vial}  Score: {score:.2f}/10.0")
                if info['remaining_after_all_ml'] < 0:
                    self.logger.warning(f"      -> WILL RUN OUT ({info['remaining_after_all_ml']:.2f} mL short)")
                else:
                    self.logger.warning(f"      -> Remaining: {info['remaining_after_all_ml']:.2f} mL, Util: {info['utilization_pct']:.1f}%")

        self.logger.info("=" * COL + "\n")
    
    def evaluate(
        self,
        vial_names: List[str],
        batch_df: pd.DataFrame,
        lash_e,
        vial_type: str = 'A'
    ):
        """Run complete risk assessment."""
        analysis = self.analyze_vial_usage(vial_names, batch_df, lash_e, vial_type)
        self.generate_risk_report(vial_names, analysis)
        return analysis


def get_vial_risk_assessment(lash_e, surfactant_shims):
    """
    Assess risk for all surfactant vials before dispense.

    Args:
        lash_e: Robot coordinator
        surfactant_shims: List of (surfactant_name, vial_list, shim_df) tuples.
            shim_df has columns 'substock_A_name' and 'surf_A_volume_ul' already
            mapped from the surfactant-specific column names.
    """
    logger = lash_e.logger
    analyzer = VialRiskAnalyzer(logger)

    logger.info("\n" + "=" * 130)
    logger.info("PRE-DISPENSE RISK ASSESSMENT")
    logger.info("=" * 130)

    for surf_name, vial_list, shim_df in surfactant_shims:
        logger.info(f"\n>>> {surf_name} Vials:")
        analyzer.evaluate(vial_list, shim_df, lash_e, vial_type='A')

    logger.info("=" * 130)
