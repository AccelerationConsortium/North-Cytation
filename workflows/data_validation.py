# data_validation.py
"""
Comprehensive data validation for calibration workflow.

This module provides validation functions to catch data integrity issues
early and prevent silent corruption of experimental results.
"""

def validate_candidate_data_integrity(candidate, context="unknown"):
    """
    Validate that a candidate has all required data fields for accurate processing.
    
    Args:
        candidate: Dictionary containing experimental result data
        context: String describing where this validation is called from
        
    Returns:
        dict: {'valid': bool, 'errors': list, 'warnings': list}
        
    Raises:
        ValueError: If critical data is missing and STRICT_VALIDATION is enabled
    """
    errors = []
    warnings = []
    
    # Required fields for any candidate
    required_fields = ['deviation', 'time', 'volume']
    
    # Check for required fields
    for field in required_fields:
        if field not in candidate:
            errors.append(f"Missing required field '{field}'")
        elif candidate[field] is None:
            errors.append(f"Required field '{field}' is None")
    
    # Validate measured volume data availability
    has_measured_volume = candidate.get('measured_volume') is not None
    has_raw_measurements = bool(candidate.get('raw_measurements', []) or candidate.get('all_measurements', []))
    has_calculated_volume = candidate.get('calculated_volume') is not None
    
    if not (has_measured_volume or has_raw_measurements or has_calculated_volume):
        errors.append("No measured volume data found - missing 'measured_volume', 'raw_measurements'/'all_measurements', AND 'calculated_volume'")
    
    # Check for data consistency
    if has_measured_volume and has_raw_measurements:
        raw_measurements = candidate.get('raw_measurements', []) or candidate.get('all_measurements', [])
        if len(raw_measurements) > 0:
            calculated_avg = sum(raw_measurements) / len(raw_measurements)
            measured_volume = candidate['measured_volume']
            
            # Allow small floating point differences
            if abs(calculated_avg - measured_volume) > 0.001:  # 1μL tolerance
                warnings.append(f"Measured volume ({measured_volume:.4f}) doesn't match raw measurements average ({calculated_avg:.4f})")
    
    # Validate parameter completeness for optimization candidates
    if candidate.get('strategy') in ['SCREENING', 'OPTIMIZATION', 'EXTERNAL_DATA']:
        required_params = ['aspirate_speed', 'dispense_speed']
        missing_params = [p for p in required_params if p not in candidate or candidate[p] is None]
        if missing_params:
            errors.append(f"Missing required parameters for {candidate.get('strategy', 'unknown')} candidate: {missing_params}")
    
    # Validate deviation range (should be reasonable for pipetting)
    if 'deviation' in candidate and candidate['deviation'] is not None:
        dev = candidate['deviation']
        if dev < 0:
            warnings.append(f"Negative deviation ({dev:.2f}%) - unusual for pipetting")
        elif dev > 200:
            warnings.append(f"Extremely high deviation ({dev:.2f}%) - possible data error")
    
    # Check for variability data
    if 'variability' not in candidate or candidate['variability'] is None:
        warnings.append("Missing variability data - precision evaluation will be limited")
    
    result = {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'context': context
    }
    
    return result

def validate_candidates_list(candidates, context="unknown"):
    """
    Validate a list of candidates and provide summary statistics.
    
    Returns:
        dict: {'all_valid': bool, 'error_count': int, 'warning_count': int, 'details': list}
    """
    if not candidates:
        return {
            'all_valid': False,
            'error_count': 1,
            'warning_count': 0,
            'details': [{'valid': False, 'errors': ['Empty candidates list'], 'warnings': [], 'context': context}]
        }
    
    all_results = []
    total_errors = 0
    total_warnings = 0
    
    for i, candidate in enumerate(candidates):
        validation_result = validate_candidate_data_integrity(candidate, f"{context}[{i}]")
        all_results.append(validation_result)
        
        if not validation_result['valid']:
            total_errors += len(validation_result['errors'])
        total_warnings += len(validation_result['warnings'])
    
    return {
        'all_valid': all(r['valid'] for r in all_results),
        'error_count': total_errors,
        'warning_count': total_warnings,
        'details': all_results
    }

def print_validation_report(validation_result, verbose=True):
    """Print a formatted validation report."""
    if validation_result['all_valid']:
        print(f"✅ DATA VALIDATION PASSED ({validation_result['context'] if 'context' in validation_result else 'unknown context'})")
        if validation_result.get('warning_count', 0) > 0:
            print(f"   ⚠️  {validation_result['warning_count']} warnings found")
            if verbose:
                for detail in validation_result.get('details', []):
                    for warning in detail.get('warnings', []):
                        print(f"      • {warning}")
    else:
        print(f"❌ DATA VALIDATION FAILED ({validation_result['context'] if 'context' in validation_result else 'unknown context'})")
        print(f"   🚨 {validation_result['error_count']} errors, {validation_result['warning_count']} warnings")
        
        if verbose:
            for detail in validation_result.get('details', []):
                if not detail['valid']:
                    print(f"   Context: {detail['context']}")
                    for error in detail['errors']:
                        print(f"      ❌ {error}")
                for warning in detail.get('warnings', []):
                    print(f"      ⚠️  {warning}")

def require_valid_candidate(candidate, context="unknown", strict=True):
    """
    Validate candidate and raise error if invalid (when strict=True).
    
    Args:
        candidate: Candidate to validate
        context: Context string for error messages
        strict: If True, raise ValueError on validation failure
        
    Returns:
        validation_result dict
        
    Raises:
        ValueError: If validation fails and strict=True
    """
    validation_result = validate_candidate_data_integrity(candidate, context)
    
    if not validation_result['valid'] and strict:
        error_msg = f"Data validation failed for {context}:\n"
        for error in validation_result['errors']:
            error_msg += f"  • {error}\n"
        error_msg += f"Available fields: {list(candidate.keys())}"
        raise ValueError(error_msg)
    
    return validation_result