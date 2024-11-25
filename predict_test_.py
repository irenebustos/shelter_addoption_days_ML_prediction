


import requests

url = 'http://localhost:9696/predict'

animal = {
 'intake_type_group': 'stray',
 'intake_condition_group': 'normal',
 'is_male': 0,
 'hair_type': 'unknown',
 'breed_group': 'border_collie',
 'color_group': 'tan & white',
 'age_in_months': 1.0,
 'month_in': 6,
 'animals_in_shelter': 944,
 'day_of_week_in': 4,
 'is_dog': 1,
 'is_mix_breed': 1,
 'is_miniature': 0,
 'is_domestic': 0,
 'is_castrated': 0}
 
response = requests.post(url, json=animal).json()

predicted_days = response.get('predicted_days_in_shelter')

if predicted_days is not None and isinstance(predicted_days, (int, float)):
    rounded_days = round(predicted_days)

    if rounded_days > 15:
        print('Estimated days the animal will spend in the shelter:', rounded_days)
        print('This is a long stay animal. Please promote this animal to be adopted.')
    else:
        print('Estimated days the animal will spend in the shelter:', rounded_days)
        print('This is a short stay animal. No need to have special attention to it.')
else:
    print("Error: Invalid response from the API. Expected a numerical prediction.")
