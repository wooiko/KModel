# test_config_params_fixed.py - Updated test script to verify fixed configuration parameter passing

import json
from config_manager import load_config, _filter_for_simulate_mpc

def test_fixed_config_parameters():
    """Test that krr-test configuration parameters are properly handled after fixes."""
    
    print("🧪 TESTING FIXED CONFIGURATION PARAMETER PASSING")
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
    
    print("\n📋 CHECKING CRITICAL PARAMETERS IN ORIGINAL CONFIG:")
    for param, expected_value in critical_params.items():
        actual_value = config.get(param)
        if actual_value == expected_value:
            print(f"   ✅ {param}: {actual_value} (correct)")
        else:
            print(f"   ❌ {param}: {actual_value} (expected {expected_value})")
    
    # Test FIXED _filter_for_simulate_mpc function
    print(f"\n🔧 TESTING FIXED _filter_for_simulate_mpc:")
    
    filtered_config = _filter_for_simulate_mpc(config)
    
    print(f"   Original config size: {len(config)} parameters")
    print(f"   Filtered config size: {len(filtered_config)} parameters")
    
    # Check if critical parameters survived filtering
    all_preserved = True
    for param, expected_value in critical_params.items():
        if param in filtered_config:
            if filtered_config[param] == expected_value:
                print(f"   ✅ {param}: {filtered_config[param]} (preserved correctly)")
            else:
                print(f"   ⚠️ {param}: {filtered_config[param]} (preserved but value changed)")
                all_preserved = False
        else:
            print(f"   ❌ {param}: MISSING after filtering")
            all_preserved = False
    
    # Test FIXED simulate_mpc parameter handling
    print(f"\n🎮 TESTING FIXED PARAMETER DEFAULTS LOGIC:")
    
    # Simulate what happens in fixed simulate_mpc
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
    original_values = {}
    for param in critical_params.keys():
        original_values[param] = test_params.get(param)
        
    for key, default_value in defaults.items():
        if key not in test_params:
            test_params[key] = default_value
    
    print("   After applying FIXED defaults (only if missing):")
    for param in ['N_data', 'λ_obj', 'model_type', 'new_param']:
        print(f"     {param}: {test_params.get(param)}")
    
    # Verify critical parameters weren't overwritten
    params_preserved = True
    for param, expected_value in critical_params.items():
        current_value = test_params.get(param)
        if current_value != expected_value:
            print(f"   ❌ {param} was altered: {current_value} != {expected_value}")
            params_preserved = False
        else:
            print(f"   ✅ {param} preserved: {current_value}")
    
    # Test what parameters would be passed to simulate_mpc
    print(f"\n📦 PARAMETERS THAT WOULD BE PASSED TO simulate_mpc:")
    key_params = ['N_data', 'model_type', 'λ_obj', 'initial_trust_radius', 'kernel', 'Np', 'Nc']
    for param in key_params:
        value = test_params.get(param, 'MISSING')
        expected = critical_params.get(param, 'N/A')
        status = "✅" if value == expected else "❌" if expected != 'N/A' else "➕"
        print(f"   {status} {param}: {value}")
    
    print(f"\n📊 OVERALL RESULT:")
    if all_preserved and params_preserved:
        print("   ✅ ALL TESTS PASSED - Configuration parameters are handled correctly")
        return True
    else:
        print("   ❌ SOME TESTS FAILED - Configuration parameters may still have issues")
        return False

def test_linear_model_support():
    """Test linear model configuration support."""
    print(f"\n🔧 TESTING LINEAR MODEL SUPPORT:")
    
    # Create test linear configuration
    linear_config = {
        'model_type': 'linear',
        'linear_type': 'ridge',
        'poly_degree': 2,
        'alpha': 0.5,
        'N_data': 2000,
        'Np': 5
    }
    
    print("   Original linear config:")
    for key, value in linear_config.items():
        print(f"     {key}: {value}")
    
    # Test filtering
    filtered = _filter_for_simulate_mpc(linear_config)
    
    print("   After filtering:")
    for key in linear_config.keys():
        if key in filtered:
            print(f"     ✅ {key}: {filtered[key]}")
        else:
            print(f"     ❌ {key}: MISSING")
    
    # Check if defaults were added properly
    expected_defaults = ['include_bias']
    for param in expected_defaults:
        if param in filtered:
            print(f"   ✅ Default added: {param} = {filtered[param]}")
        else:
            print(f"   ❌ Default missing: {param}")

if __name__ == '__main__':
    success = test_fixed_config_parameters()
    test_linear_model_support()
    
    if success:
        print(f"\n🎉 CONFIGURATION PARAMETER PASSING IS NOW WORKING CORRECTLY!")
    else:
        print(f"\n⚠️ CONFIGURATION PARAMETER PASSING STILL HAS ISSUES!")