"""Reduce linear protocol constraints with fixed parameters to one interval.

Accepts the arithmetic inequalities used by protocol get_parameter_constraints.
Unsupported expressions and violated fixed-only constraints fail explicitly.
No hardware names, capacities, or executable expressions are embedded here.
"""
import ast
import math
import logging

logger = logging.getLogger(__name__)


def constrain_parameters(parameters, constraints, context=None):
    """Bound a prepared parameter mapping after rounding or compensation.

    All parameters except the calibration correction remain fixed. The caller
    supplies protocol inequalities and any non-parameter expression context.
    """
    fixed_values = {**parameters, **(context or {})}
    lower, upper = feasible_interval(constraints, fixed_values, 'overaspirate_vol')
    original = float(parameters['overaspirate_vol'])
    if not math.isfinite(original):
        raise ValueError('Non-finite overaspirate_vol')
    bounded = min(upper, max(lower, original))
    if bounded != original:
        logger.info('Bounding prepared overaspirate_vol from %.9f to %.9f using protocol constraints', original, bounded)
    return {**parameters, 'overaspirate_vol': bounded}


def feasible_interval(constraints, fixed_values, variable, lower=-math.inf, upper=math.inf):
    """Intersect inequalities after substituting every parameter but variable."""
    def linear(node):
        # Represent each expression as coefficient * variable + constant.
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if math.isfinite(value):
                return 0.0, value
        if isinstance(node, ast.Name):
            if node.id == variable:
                return 1.0, 0.0
            if node.id not in fixed_values:
                raise ValueError(f"Missing fixed constraint parameter: {node.id}")
            value = float(fixed_values[node.id])
            if math.isfinite(value):
                return 0.0, value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            coefficient, constant = linear(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return sign * coefficient, sign * constant
        if isinstance(node, ast.BinOp):
            a, b = linear(node.left)
            c, d = linear(node.right)
            if isinstance(node.op, ast.Add):
                return a + c, b + d
            if isinstance(node.op, ast.Sub):
                return a - c, b - d
            if isinstance(node.op, ast.Mult) and (a == 0 or c == 0):
                return a * d + b * c, b * d
            if isinstance(node.op, ast.Div) and c == 0 and d != 0:
                return a / d, b / d
        raise ValueError("Constraints must contain finite linear arithmetic only")

    for expression in constraints:
        try:
            node = ast.parse(expression, mode='eval').body
        except SyntaxError as error:
            raise ValueError(f"Invalid constraint: {expression}") from error
        if (not isinstance(node, ast.Compare) or len(node.ops) != 1
                or not isinstance(node.ops[0], (ast.LtE, ast.GtE))):
            raise ValueError(f"Unsupported constraint: {expression}")
        a, b = linear(node.left)
        c, d = linear(node.comparators[0])
        coefficient, bound = a - c, d - b
        if isinstance(node.ops[0], ast.GtE):
            coefficient, bound = -coefficient, -bound
        if not math.isfinite(coefficient) or not math.isfinite(bound):
            raise ValueError(f"Non-finite constraint: {expression}")
        if coefficient > 0:
            upper = min(upper, bound / coefficient)
        elif coefficient < 0:
            lower = max(lower, bound / coefficient)
        elif bound < -1e-12:
            raise ValueError(f"Fixed parameters violate constraint: {expression}")
    if lower > upper:
        raise ValueError(f"No feasible interval for {variable}: [{lower}, {upper}]")
    return lower, upper


def distinct_bounded_point(baseline, proposed, lower, upper, minimum_spread):
    """Bound a probe, reversing direction when the preferred side has no room."""
    point = min(upper, max(lower, proposed))
    if abs(point - baseline) >= minimum_spread - 1e-12:
        return point
    spread = max(abs(proposed - baseline), minimum_spread)
    alternative = baseline - spread if proposed >= baseline else baseline + spread
    point = min(upper, max(lower, alternative))
    if abs(point - baseline) < minimum_spread - 1e-12:
        raise ValueError("Protocol constraints leave insufficient room for two distinct calibration points")
    return point
