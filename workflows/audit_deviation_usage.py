# audit_deviation_usage.py
"""
Audit script to verify that deviation is ONLY used for optimization objectives,
never for volume calculations that could corrupt measured data.
"""

import re
import os

def audit_deviation_usage(file_path):
    """
    Audit a Python file for problematic deviation usage patterns.
    
    Returns dict with findings categorized as 'dangerous', 'acceptable', 'suspicious'
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    findings = {
        "dangerous": [],      # Patterns that definitely corrupt data
        "suspicious": [],     # Patterns that need review
        "acceptable": []      # Legitimate uses of deviation
    }
    
    # Dangerous patterns - deviation used to calculate measured volumes
    dangerous_patterns = [
        (r'measured.*=.*target.*\*.*\(1.*[-+].*deviation', "Measured volume calculated from deviation"),
        (r'volume.*=.*target.*\*.*\(1.*[-+].*dev', "Volume calculated from deviation"),  
        (r'calculated.*=.*\*.*\(1.*[-+].*deviation', "Calculated volume from deviation"),
        (r'measured.*target.*\(1\s*[-+]\s*deviation', "Measured volume reconstruction"),
        (r'under_delivery.*=.*target.*\*.*\(1.*-.*deviation', "Under-delivery assumption"),
        (r'over_delivery.*=.*target.*\*.*\(1.*\+.*deviation', "Over-delivery assumption")
    ]
    
    # Suspicious patterns - might be problematic
    suspicious_patterns = [
        (r'deviation.*\*.*target', "Deviation multiplied by target"),
        (r'target.*\*.*deviation', "Target multiplied by deviation"),
        (r'fallback.*deviation', "Fallback involving deviation"),
        (r'reconstruct.*deviation', "Reconstruction from deviation"),
        (r'assume.*delivery', "Delivery direction assumption")
    ]
    
    # Acceptable patterns - legitimate uses
    acceptable_patterns = [
        (r'deviation_ul.*=.*deviation.*\*.*target', "Converting deviation % to μL for tolerance checking"),
        (r'deviation.*=.*abs\(.*measured.*-.*target\)', "Calculating deviation from measurements"),
        (r'shortfall.*=.*target.*-.*measured', "Calculating shortfall from actual measurements"),
        (r'print.*deviation', "Logging/printing deviation values"),
        (r'best.*deviation', "Selecting best by deviation"),
        (r'deviation.*<', "Deviation threshold checking"),
        (r'deviation.*>', "Deviation threshold checking")
    ]
    
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        
        # Skip comments and docstrings
        if line_lower.startswith('#') or '"""' in line or "'''" in line:
            continue
            
        # Check dangerous patterns first
        for pattern, description in dangerous_patterns:
            if re.search(pattern, line_lower):
                findings["dangerous"].append({
                    "line": i,
                    "content": line.strip(),
                    "issue": description,
                    "pattern": pattern
                })
                break  # Don't double-count
        else:
            # Check suspicious patterns
            for pattern, description in suspicious_patterns:
                if re.search(pattern, line_lower):
                    findings["suspicious"].append({
                        "line": i,
                        "content": line.strip(),
                        "issue": description,
                        "pattern": pattern
                    })
                    break
            else:
                # Check acceptable patterns  
                for pattern, description in acceptable_patterns:
                    if re.search(pattern, line_lower):
                        findings["acceptable"].append({
                            "line": i,
                            "content": line.strip(),
                            "use": description,
                            "pattern": pattern
                        })
                        break
    
    return findings

def print_audit_report(findings, file_path):
    """Print formatted audit report."""
    print(f"\n🔍 DEVIATION USAGE AUDIT: {os.path.basename(file_path)}")
    print("=" * 60)
    
    # Dangerous findings
    if findings.get("dangerous"):
        print(f"\n❌ DANGEROUS PATTERNS ({len(findings['dangerous'])} found):")
        for finding in findings["dangerous"]:
            print(f"   Line {finding['line']:3d}: {finding['issue']}")
            print(f"            Code: {finding['content']}")
    else:
        print(f"\n✅ No dangerous deviation patterns found!")
    
    # Suspicious findings  
    if findings.get("suspicious"):
        print(f"\n⚠️  SUSPICIOUS PATTERNS ({len(findings['suspicious'])} found - review needed):")
        for finding in findings["suspicious"]:
            print(f"   Line {finding['line']:3d}: {finding['issue']}")
            print(f"            Code: {finding['content']}")
    else:
        print(f"\n✅ No suspicious patterns found!")
    
    # Acceptable findings
    if findings.get("acceptable"):
        print(f"\n✅ ACCEPTABLE USES ({len(findings['acceptable'])} found):")
        for finding in findings["acceptable"][:5]:  # Show first 5
            print(f"   Line {finding['line']:3d}: {finding['use']}")
        if len(findings["acceptable"]) > 5:
            print(f"   ... and {len(findings['acceptable']) - 5} more acceptable uses")
    
    # Summary
    dangerous_count = len(findings.get("dangerous", []))
    suspicious_count = len(findings.get("suspicious", []))
    
    print(f"\n📊 SUMMARY:")
    if dangerous_count == 0 and suspicious_count == 0:
        print(f"   🎯 AUDIT PASSED - No problematic deviation usage found!")
    else:
        print(f"   🚨 ISSUES FOUND: {dangerous_count} dangerous, {suspicious_count} suspicious")
        print(f"   💡 Review and fix dangerous patterns before using in production")

if __name__ == "__main__":
    # Audit main calibration file
    main_file = "calibration_sdl_simplified.py"
    findings = audit_deviation_usage(main_file)
    print_audit_report(findings, main_file)