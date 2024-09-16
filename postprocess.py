import pandas as pd

# Define the entity_unit_map and allowed_units
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for units in entity_unit_map.values() for unit in units}

# Function to parse the prediction and check the unit
def validate_prediction(prediction):
    if not isinstance(prediction, str) or prediction.strip() == "":
        return ""
    
    # Regex pattern to check for range (e.g., "[123, 140] volt")
    range_pattern = re.compile(r'\[(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)\]\s+([a-zA-Z\s]+)')
    range_match = range_pattern.match(prediction.strip())
    
    if range_match:
        # Extract numbers and unit
        num1, _, num2, _, unit = range_match.groups()
        unit = unit.strip().lower()
        
        # Choose the higher value
        higher_value = max(float(num1), float(num2))
        
        # Validate the unit
        if unit in allowed_units:
            return ""
            # return f"{higher_value} {unit}"
        else:
            return ""
    
    # Regex pattern to match a single number followed by a unit
    single_value_pattern = re.compile(r'^-?\d+(\.\d+)?\s+([a-zA-Z\s]+)$')
    match = single_value_pattern.match(prediction.strip())
    
    if not match:
        return ""
    
    number, unit = match.groups()
    unit = unit.strip().lower()
    
    # Check if the unit is in the allowed_units set
    if unit in allowed_units:
        return prediction  # Valid prediction, return as is
    else:
        return ""  # Invalid unit, replace with empty string


df = pd.read_csv("output.csv")

# Apply the validation function to the 'prediction' column
df['prediction'] = df['prediction'].apply(validate_prediction)

# Save the updated DataFrame to a new CSV file
df.to_csv('submission.csv', index=True)