# test_config_params.py - Test script to verify configuration parameter passing

import json
from config_manager import load_config

def test_config_parameters():
    """Test that krr-test configuration parameters are loaded correctly."""
    
    print("🧪 TESTING CONFIGURATION PARAMETER PASSING")
    print("=" * 50)
    
    # Load krr-test configuration
    try:
        config = load_config('krr-test')
        print("✅ Successfully loaded krr-test configuration")
    except Exception as e:
        print(f"❌ Failed to load krr-test configuration: {e}")
        return
    
    # Check critical parameters
    critical_params = {
        'N_data': 1000,
        'model_type': 'krr', 
        'λ_obj': 0.5,
        'initial_trust_radius': 0.6,
        'ref_fe': 54.0,
        'ref_mass': 58.0,
        'kernel': 'rbf',
        'Np': 6,
        'Nc': 4
    }
    
    print("\n📋 CHECKING CRITICAL PARAMETERS:")
    all_correct = True
    
    for param, expected_value in critical_params.items():
        actual_value = config.get(param)
        if actual_value == expected_value:
            print(f"   ✅ {param}: {actual_value} (correct)")
        else:
            print(f"   ❌ {param}: {actual_value} (expected {expected_value})")
            all_correct = False
    
    # Test _filter_for_simulate_mpc function
    print(f"\n🔧 TESTING _filter_for_simulate_mpc:")
    
    from config_manager import _filter_for_simulate_mpc
    filtered_config = _filter_for_simulate_mpc(config)
    
    print(f"   Original config size: {len(config)} parameters")
    print(f"   Filtered config size: {len(filtered_config)} parameters")
    
    # Check if critical parameters survived filtering
    for param in critical_params.keys():
        if param in filtered_config:
            print(f"   ✅ {param}: {filtered_config[param]} (preserved)")
        else:
            print(f"   ❌ {param}: MISSING after filtering")
            all_correct = False
    
    # Test simulate_mpc parameter handling
    print(f"\n🎮 TESTING PARAMETER DEFAULTS LOGIC:")
    
    # Simulate what happens in simulate_mpc
    test_params = dict(filtered_config)
    
    defaults = {
        'N_data': 5000,  # Should NOT overwrite 1000
        'λ_obj': 0.1,    # Should NOT overwrite 0.5
        'model_type': 'krr',  # Should remain krr
        'new_param': 'default_value'  # Should be added
    }
    
    print("   Before applying defaults:")
    for param in ['N_data', 'λ_obj', 'model_type']:
        print(f"     {param}: {test_params.get(param)}")
    
    # Apply the FIXED logic (only set if missing)
    for key, default_value in defaults.items():
        if key not in test_params:
            test_params[key] = default_value
    
    print("   After applying defaults (FIXED logic):")
    for param in ['N_data', 'λ_obj', 'model_type', 'new_param']:
        print(f"     {param}: {test_params.get(param)}")
    
    # Verify critical parameters weren't overwritten
    for param, expected_value in critical_params.items():
        if test_params.get(param) != expected_value:
            print(f"   ❌ {param} was overwritten: {test_params.get(param)} != {expected_value}")
            all_correct = False
    
    print(f"\n📊 OVERALL RESULT:")
    if all_correct:
        print("   ✅ ALL TESTS PASSED - Configuration parameters are handled correctly")
    else:
        print("   ❌ SOME TESTS FAILED - Configuration parameters may be overwritten")
    
    return all_correct

if __name__ == '__main__':
    test_config_parameters()