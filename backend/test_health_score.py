def test_health_score_calculation():
    """Test the health score calculation with realistic values"""
    
    # Test data with realistic lab values (all should be normal)
    test_data = {
        'HAEMATOCRIT': 41.5,    # Normal range: 36-46%
        'HAEMOGLOBINS': 13.8,   # Normal range: 12-16 g/dL
        'ERYTHROCYTE': 4.6,     # Normal range: 4.0-5.5 × 10¹²/L
        'LEUCOCYTE': 8.2,       # Normal range: 4.0-11.0 × 10⁹/L
        'THROMBOCYTE': 320,     # Normal range: 150-450 × 10⁹/L
        'MCH': 30.2,            # Normal range: 27-33 pg
        'MCHC': 33.5,           # Normal range: 32-36 g/dL
        'MCV': 92.1,            # Normal range: 80-100 fL
    }
    
    # Health score calculation (same as frontend)
    parameters = [
        ('LEUCOCYTE', test_data['LEUCOCYTE'], 4.0, 11.0),
        ('THROMBOCYTE', test_data['THROMBOCYTE'], 150, 450),
        ('MCH', test_data['MCH'], 27, 33),
        ('MCHC', test_data['MCHC'], 32, 36),
        ('MCV', test_data['MCV'], 80, 100),
        ('HAEMATOCRIT', test_data['HAEMATOCRIT'], 36, 46),
        ('HAEMOGLOBINS', test_data['HAEMOGLOBINS'], 12, 16),
        ('ERYTHROCYTE', test_data['ERYTHROCYTE'], 4.0, 5.5)
    ]
    
    score = 0
    total = len(parameters)
    
    print("🏥 Health Score Calculation Test:")
    print("=" * 50)
    
    for param_name, value, min_val, max_val in parameters:
        status = "NORMAL" if min_val <= value <= max_val else "ABNORMAL"
        print(f"  {param_name}: {value} ({status}) - Range: {min_val}-{max_val}")
        if min_val <= value <= max_val:
            score += 1
    
    expected_score = round((score / total) * 100)
    print(f"\n🎯 Health Score: {expected_score}% ({score}/{total} parameters normal)")
    
    if expected_score == 100:
        print("✅ All parameters are within normal ranges!")
    else:
        print(f"⚠️ {total - score} parameters are outside normal ranges")
    
    return expected_score

if __name__ == "__main__":
    test_health_score_calculation() 