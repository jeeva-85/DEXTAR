import urllib.request
import json

scenarios = [
    {
        "name": "Scenario 1: Healthy / Normal Railway Condition",
        "data": {
            "train_age_years": 4.0,
            "average_speed_kmph": 95.0,
            "distance_travelled_km": 35000.0,
            "ambient_temperature_c": 26.0,
            "humidity_percent": 50.0,
            "rainfall_mm": 0.0,
            "wheel_wear_percent": 12.0,
            "track_vibration_level": 1.2,
            "rail_wear_mm": 4.2,
            "bearing_temperature_c": 52.0,
            "axle_temperature_c": 48.0,
            "brake_pad_wear_percent": 15.0,
            "brake_pressure_psi": 98.0,
            "battery_voltage": 25.4,
            "last_maintenance_days": 18.0,
            "sensor_health_index": 94.0,
            "inspection_score": 92.0,
            "delay_minutes": 0.0,
            "region": "Northern Railway",
            "season": "Winter",
            "train_type": "Express",
        },
    },
    {
        "name": "Scenario 2: Moderate-Risk Railway Condition",
        "data": {
            "train_age_years": 14.0,
            "average_speed_kmph": 78.0,
            "distance_travelled_km": 95000.0,
            "ambient_temperature_c": 30.0,
            "humidity_percent": 62.0,
            "rainfall_mm": 8.0,
            "wheel_wear_percent": 46.0,
            "track_vibration_level": 3.4,
            "rail_wear_mm": 9.5,
            "bearing_temperature_c": 76.0,
            "axle_temperature_c": 66.0,
            "brake_pad_wear_percent": 48.0,
            "brake_pressure_psi": 92.0,
            "battery_voltage": 24.1,
            "last_maintenance_days": 140.0,
            "sensor_health_index": 68.0,
            "inspection_score": 65.0,
            "delay_minutes": 14.0,
            "region": "Western Railway",
            "season": "Summer",
            "train_type": "Passenger",
        },
    },
    {
        "name": "Scenario 3: High-Risk / Critical Failure Condition",
        "data": {
            "train_age_years": 26.0,
            "average_speed_kmph": 65.0,
            "distance_travelled_km": 180000.0,
            "ambient_temperature_c": 38.0,
            "humidity_percent": 78.0,
            "rainfall_mm": 35.0,
            "wheel_wear_percent": 85.0,
            "track_vibration_level": 6.2,
            "rail_wear_mm": 15.0,
            "bearing_temperature_c": 98.0,
            "axle_temperature_c": 84.0,
            "brake_pad_wear_percent": 82.0,
            "brake_pressure_psi": 80.0,
            "battery_voltage": 21.8,
            "last_maintenance_days": 310.0,
            "sensor_health_index": 32.0,
            "inspection_score": 38.0,
            "delay_minutes": 55.0,
            "region": "Eastern Railway",
            "season": "Monsoon",
            "train_type": "Freight",
        },
    },
]

results = []
for s in scenarios:
    req = urllib.request.Request(
        "http://localhost:8000/api/ml/predict",
        data=json.dumps(s["data"]).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        out = json.loads(resp.read().decode())
        res_item = {
            "scenario": s["name"],
            "priority_score": out["priority"]["score"],
            "priority_level": out["priority"]["level"],
            "failure_probability": out["failure_risk"]["probability"],
            "failure_risk_level": out["failure_risk"]["level"],
            "predicted_severity": out["failure_severity"]["predicted_severity"],
            "top_shap_factors": out["shap_explanation"]["top_factors"][:4],
        }
        results.append(res_item)
        print(f"=== {s['name']} ===")
        print(f"Priority Score: {res_item['priority_score']} ({res_item['priority_level']})")
        print(f"Failure Probability: {res_item['failure_probability']} ({res_item['failure_risk_level']})")
        print(f"Severity: {res_item['predicted_severity']}")
        print("Top SHAP Factors:")
        for factor in res_item["top_shap_factors"]:
            print(f"  * {factor['feature']}: {factor['shap_value']:+.4f} ({factor['direction']})")
        print()

with open("test_scenarios_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved output to test_scenarios_output.json")
