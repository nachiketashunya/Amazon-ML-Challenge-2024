import pandas as pd
import re

df_train = pd.read_csv('train.csv')

# Define the accepted units
accepted_units = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

def process_prediction(value):
    # Function to process a single prediction value
    if pd.isna(value):
        return "NA"
    
    # Convert to string
    value = str(value)
    
    # Check if it's a range value
    range_match = re.match(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]\s*(\w+)', value)
    if range_match:
        return f"{range_match.group(2)} {range_match.group(3)}"
    
    # Split the value into numeric part and unit
    parts = value.split()
    if len(parts) != 2:
        return "NA"
    
    numeric, unit = parts
    
    # Check if the unit is in any of the accepted unit sets
    if not any(unit in unit_set for unit_set in accepted_units.values()):
        return "NA"
    
    return value

def process_dataframe(df):
    # Apply the processing function to the 'prediction' column
    df['entity_value'] = df['entity_value'].apply(process_prediction)
    return df

processed_df = process_dataframe(df_train)

processed_df.to_csv("data/processed_train.csv",index = False)