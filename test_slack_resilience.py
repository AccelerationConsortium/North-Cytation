#!/usr/bin/env python3
"""
Test script to verify Slack error handling doesn't crash workflows.
"""

import slack_agent
import time

def test_slack_resilience():
    """Test that Slack failures don't crash the workflow."""
    print("=== Testing Slack Resilience ===")
    
    # Test 1: Check connectivity
    print("\n1. Testing connectivity...")
    is_connected = slack_agent.test_slack_connectivity()
    print(f"Slack connectivity: {'✅ Connected' if is_connected else '❌ No connection'}")
    
    # Test 2: Try sending a message (should handle errors gracefully)
    print("\n2. Testing message sending...")
    success = slack_agent.send_slack_message("Test message - workflow resilience check")
    print(f"Message send result: {'✅ Success' if success else '❌ Failed (but workflow continues)'}")
    
    # Test 3: Try safe wrapper
    print("\n3. Testing safe wrapper...")
    safe_success = slack_agent.safe_send_slack_message("Safe test message", silent_fail=False)
    print(f"Safe send result: {'✅ Success' if safe_success else '❌ Failed (but workflow continues)'}")
    
    # Test 4: Simulate workflow continuation
    print("\n4. Simulating workflow continuation...")
    for i in range(3):
        print(f"Workflow step {i+1}...")
        slack_agent.safe_send_slack_message(f"Workflow step {i+1} completed")
        time.sleep(0.5)
    
    print("\n✅ Workflow completed successfully despite any Slack issues!")
    print("The workflow will continue even if Slack is unavailable.")

if __name__ == "__main__":
    test_slack_resilience()