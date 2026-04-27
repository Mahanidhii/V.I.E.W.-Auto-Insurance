import numpy as np

# Actuarial Constants & Rules 

# Map VehiDE vision classes to Insurance Severities
DAMAGE_SEVERITY_MAP = {
    "rach": "Minor",           # Scratch/Paint damage
    "mop_lom": "Moderate",     # Broken Lamp/Glass
    "be_den": "Moderate",      # Dent
    "tray_son": "Severe"       # Tear/Severe structural
}

# Realistic repair cost baselines (INR) calibrated to Indian market:
# Minor (scratch/paint): ₹1,500–₹6,000
# Moderate (dent/glass): ₹8,000–₹25,000
# Severe (structural):   ₹40,000–₹1,20,000
BASE_SEVERITY_COST = {
    "Minor":    3500,      # Paint scratch / minor surface damage
    "Moderate": 18000,     # Dent, broken lamp, glass
    "Severe":   65000      # Structural / tear damage
}

BRAND_TIERS = {
    "Budget":    ["Maruti", "Tata", "Mahindra", "Hyundai", "Renault", "Datsun"],
    "Mid-Range": ["Honda", "Toyota", "Kia", "MG", "Skoda", "Volkswagen", "Jeep", "Ford"],
    "Premium":   ["BMW", "Mercedes-Benz", "Audi", "Volvo", "Land Rover", "Jaguar", "Porsche"]
}

BRAND_MULTIPLIERS = {
    "Budget":    1.0,
    "Mid-Range": 1.4,
    "Premium":   2.2
}

FUEL_MULTIPLIERS = {
    "Petrol":   1.0,
    "Diesel":   1.05,
    "Hybrid":   1.25,
    "Electric": 1.50
}

# collision severity, independent of visual damage class
COLLISION_SEVERITY_MULTIPLIERS = {
    "Minor Collision":    0.65,   # Low-speed bump / nudge
    "Moderate Collision": 1.00,   # Standard mid-speed impact
    "Major Collision":    1.55    # High-speed / severe impact
}

# Core Logic Functions
def get_brand_tier(auto_make):
    """Determines the tier of the car brand."""
    for tier, brands in BRAND_TIERS.items():
        if auto_make in brands:
            return tier
    return "Mid-Range"  # Fallback for unknown brands

def calculate_depreciation_multiplier(vehicle_age):
    """Calculates depreciation: 8% per year, capped at 60%."""
    depreciation_pct = min(vehicle_age * 0.08, 0.60)
    return 1.0 - depreciation_pct

def calculate_rule_based_estimate(auto_make, vehicle_age, fuel_type,
                                   detected_class, collision_severity):
    """
    Calculates the strict actuarial fallback estimate.

    Formula:
        Base_Cost (by visual damage)
        × Brand_Multiplier
        × Fuel_Multiplier
        × Collision_Severity_Multiplier
        × Depreciation_Multiplier
    """
    # Base Cost from visual damage class
    severity = DAMAGE_SEVERITY_MAP.get(detected_class, "Moderate")
    base_cost = BASE_SEVERITY_COST[severity]

    # Multipliers
    tier         = get_brand_tier(auto_make)
    brand_mult   = BRAND_MULTIPLIERS[tier]
    fuel_mult    = FUEL_MULTIPLIERS.get(fuel_type, 1.0)
    col_mult     = COLLISION_SEVERITY_MULTIPLIERS.get(collision_severity, 1.0)
    dep_mult     = calculate_depreciation_multiplier(vehicle_age)

    # Final Rule-Based calculation
    final_estimate = base_cost * brand_mult * fuel_mult * col_mult * dep_mult

    # Floor at ₹1,000 (absolute minimum realistic repair)
    return max(final_estimate, 1000.0)


def generate_hybrid_claim(ml_prediction, auto_make, vehicle_age, fuel_type,
                           detected_class, collision_severity="Moderate Collision"):
    """
    Blends the ML prediction with the rule-based estimate for safety.

    If the ML prediction is invalid (negative, zero, or >3× the rule base),
    the rule-based estimate is used exclusively.  Otherwise a weighted
    blend (60 % ML + 40 % rules) is returned.
    """
    rule_estimate = calculate_rule_based_estimate(
        auto_make, vehicle_age, fuel_type, detected_class, collision_severity
    )

    if ml_prediction <= 0 or ml_prediction > (rule_estimate * 3):
        final_claim = rule_estimate
        method_used = "Rule-Based Fallback (ML Anomaly Detected)"
    else:
        # 60 % ML, 40 % rules — rules act as a realistic anchor
        final_claim = (ml_prediction * 0.60) + (rule_estimate * 0.40)
        method_used = "Hybrid Ensemble (CatBoost + Actuarial Rules)"

    # ± 15 % confidence band
    low_range  = final_claim * 0.85
    high_range = final_claim * 1.15

    severity = DAMAGE_SEVERITY_MAP.get(detected_class, "Moderate")
    return {
        "final_claim": round(final_claim, 2),
        "low_range":   round(low_range, 2),
        "high_range":  round(high_range, 2),
        "method_used": method_used,
        "breakdown": {
            "base_severity":       BASE_SEVERITY_COST[severity],
            "brand_multiplier":    BRAND_MULTIPLIERS[get_brand_tier(auto_make)],
            "fuel_multiplier":     FUEL_MULTIPLIERS.get(fuel_type, 1.0),
            "collision_multiplier": COLLISION_SEVERITY_MULTIPLIERS.get(collision_severity, 1.0),
            "depreciation_factor": calculate_depreciation_multiplier(vehicle_age),
        }
    }