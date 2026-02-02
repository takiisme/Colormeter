from util import load_data
from cross_validation import run_comprehensive_cross_validation, run_leave_one_out_analysis



# Train the model and also apply correction by scaling


# Cross validation
df_daylight1 = load_data("Data/daylight1.json")
df_daylight1["lighting_condition"] = "daylight"
df_daylight2 = load_data("Data/daylight2.json")
df_daylight2["lighting_condition"] = "daylight"
df_raw = pd.concat([df_daylight1, df_daylight2], ignore_index=True)
run_comprehensive_cross_validation(df_raw)
run_leave_one_out_analysis(df_raw)
