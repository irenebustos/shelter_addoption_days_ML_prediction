import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
import xgboost as xgb
from collections import Counter  # If used for word frequency analysis
from itertools import combinations  # If combinations are still relevant
import warnings 

# Ignore warnings 
warnings.filterwarnings("ignore", category=UserWarning)

# Parameters
percentage_limit_breed_words = 80
percentage_limit_breed_words_combination = 0.8  
percentage_limit_breed_words_unique = 2 
percentage_limit_color = 0.05  

eta = 0.1
max_depth = 6
gamma = 0.1
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 15
nthread = -1
seed = 1
verbosity = 1
eval_metric = 'rmse'
output_file = f'model_eta={eta}_maxdepth={max_depth}_gamma={gamma}_subsample={subsample}_colsample_bytree={colsample_bytree}_minchild={min_child_weight}.bin'

#import the data 
url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

#####################
#####################
# data preparation
datetime_columns = ['datetime_intake', 'datetime_outcome']
# convert the datetime columns to datetime type
for column in datetime_columns:
    df[column] = pd.to_datetime(df[column])
datetime_columns

# rest of the columns from dataframe that are not datetime categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns

for column in categorical_columns:
    df[column] = df[column].str.lower()

##########################
# days in shelter
df['days_in_shelter'] = (df['datetime_outcome'] - df['datetime_intake']).dt.days

##########################
# castrated
#########################
def castrated_status(row):
    if 'neutered' in row or 'spayed' in row:
        return 'yes'
    else:
        return 'no'

df['castrated'] = df['sex_upon_intake'].apply(castrated_status)
df['is_castrated'] = df['castrated'].apply(lambda x: 1 if x =='yes' else 0)

##########################
# sex upon intake
##########################
def sex_upon_intake(row):
    if 'female' in row:
        return 'female'
    elif 'male' in row:
        return 'male'
    else:
        return 'unknown'

df['sex_upon_intake'] = df['sex_upon_intake'].apply(sex_upon_intake) 

#############################################
# Age in months when the animal was taken in
#############################################
def convert_to_months(age):
    # Split the age into value and unit
    parts = age.split()
    if len(parts) != 2:  # Handle unexpected formats
        return None
    
    value, unit = int(parts[0]), parts[1].lower()
    
    # Convert the age to months
    if 'year' in unit:
        return value * 12
    elif 'month' in unit:
        return value 
    elif 'week' in unit:
        return value / 4
    elif 'day' in unit:
        return value / 30
    else:
        return None  # Handle unknown units

df['age_in_months'] = df['age_upon_intake'].apply(convert_to_months) 
df['age_in_months'] = df['age_in_months'].abs()

##########################
# Breed group creation 
##########################
# Hair type
df['hair_type'] = df['breed'].apply(
    lambda x: 'long' if 'longhair' in x else 'short' if 'shorthair' in x
    else 'medium' if 'medium hair' in x else 'unknown'
)

# Mix breed
df['mix_breed'] = df['breed'].apply(lambda x: 'mix' if 'mix' in x else 'not mix')
df['is_mix_breed'] = df['mix_breed'].apply(lambda x: 1 if x == 'mix' else 0)

# Miniature breed
df['miniature'] = df['breed'].apply(lambda x: 'miniature' if 'miniature' in x else 'non-miniature')
df['is_miniature'] = df['miniature'].apply(lambda x: 1 if x == 'miniature' else 0)

# Domestic breed
df['domestic'] = df['breed'].apply(lambda x: 'domestic' if 'domestic' in x else 'non-domestic')
df['is_domestic'] = df['domestic'].apply(lambda x: 1 if x == 'domestic' else 0)

# Clean up breed column by removing specific words
words_to_remove = ['mix', 'shorthair', 'longhair', 'medium hair', 'miniature', 'domestic', 'dog', 'cat']
for word in words_to_remove:
    df['breed'] = df['breed'].str.replace(word, '', regex=True).str.strip()

# Tokenize and count words in the breed column
all_words = df['breed'].str.split(expand=True).stack()
word_counts = Counter(all_words)
word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Calculate percentage and cumulative percentage of word frequencies
total_words = word_freq_df['Frequency'].sum()
word_freq_df['Percentage'] = (word_freq_df['Frequency'] / total_words) * 100
word_freq_df['Cumulative Percentage'] = word_freq_df['Percentage'].cumsum()

# Select important words based on cumulative percentage limit
important_words_df = word_freq_df[word_freq_df['Cumulative Percentage'] <= percentage_limit_breed_words]

# Generate pairs of words in the breed column
df['breed_tokenized'] = df['breed'].str.split()
word_pairs = df['breed_tokenized'].apply(lambda x: list(combinations(x, 2)))

all_pairs = [pair for pairs in word_pairs for pair in pairs]
pair_counts = Counter(all_pairs)
pair_freq_df = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Calculate percentage and cumulative percentage of pair frequencies
total_pairs = pair_freq_df['Frequency'].sum()
pair_freq_df['Percentage'] = (pair_freq_df['Frequency'] / total_pairs) * 100
pair_freq_df['Cumulative Percentage'] = pair_freq_df['Percentage'].cumsum()

# Select frequent pairs based on percentage limit
frequent_pairs_df = pair_freq_df[pair_freq_df['Percentage'] >= percentage_limit_breed_words_combination]

# Function to assign a breed group based on frequent pairs
def assign_breed_group(breed, frequent_pairs):
    for pair in frequent_pairs:
        if all(word in breed for word in pair):
            return f"{pair[0]}_{pair[1]}"
    return None

df['breed_group1'] = df['breed'].apply(lambda x: assign_breed_group(x, frequent_pairs_df['Pair'].tolist()))

# Clean up the breed column by removing frequent pairs
top_combinations = frequent_pairs_df['Pair'].tolist()

def remove_combinations(breed, combinations):
    for pair in combinations:
        if all(word in breed for word in pair):
            breed = breed.replace(f"{pair[0]} {pair[1]}", "")
    return breed.strip()

df['breed'] = df['breed'].apply(lambda x: remove_combinations(x, top_combinations))

# Recalculate frequent words after cleaning the breed column
df_no_breed_group = df[df['breed_group1'].isna()]
all_words = df_no_breed_group['breed'].str.split(expand=True).stack()
word_counts = Counter(all_words)

word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Select important words based on frequency percentage limit
total_words = word_freq_df['Frequency'].sum()
word_freq_df['Percentage'] = (word_freq_df['Frequency'] / total_words) * 100
word_freq_df['Cumulative Percentage'] = word_freq_df['Percentage'].cumsum()
frequent_words_df = word_freq_df[word_freq_df['Percentage'] >= percentage_limit_breed_words_unique]

# Exclude specific words from the frequent words
exclude_words = ['bull', 'pit']
frequent_words_df = frequent_words_df[~frequent_words_df['Word'].isin(exclude_words)]

# Function to assign breed group based on frequent words
def assign_breed_word(breed, frequent_words):
    for word in frequent_words:
        if word in breed:
            return word
    return None

df['breed_group2'] = df['breed'].apply(lambda x: assign_breed_word(x, frequent_words_df['Word'].tolist()))

# Combine breed groups into a single column
df['breed_group'] = df['breed_group1'].fillna(df['breed_group2']).fillna('Other')

# Group breeds into specific categories
df['breed_group'] = df['breed_group'].apply(
    lambda x: 'larger_dangerous' if 'pit' in x or 'bull' in x or 'american_terrier' in x
    else 'small_dog' if 'chihuahua' in x or 'terrier' in x or 'dachshund' in x or 'poodle' in x or 'jack_russell' in x or 'russell_terrier' in x
    else x
)
##########################
# color 
#########################
# Split the color column into components
color_combinations = df['color'].str.split('/')  # Split by '/'
split_colors = color_combinations.apply(lambda x: [part.split()[0] for part in x] if isinstance(x, list) else [])  # Extract first word of each component

# Extract primary and secondary colors
df['color_primary'] = split_colors.apply(lambda x: x[0] if len(x) > 0 else None)  # First color
df['color_secondary'] = split_colors.apply(lambda x: x[1] if len(x) > 1 else None)  # Second color

# Identify dominant single-color groups 
single_colors = df[df['color_secondary'].isnull()]['color_primary']  # Single-color records
single_color_counts = single_colors.value_counts()
total_single = single_colors.count()
dominant_single_colors = single_color_counts[single_color_counts / total_single > percentage_limit_color].index

df['single_color_group'] = df['color_primary'].apply(
    lambda x: x if x in dominant_single_colors else 'other_single_colour'
)

# Process two-color combinations
df['sorted_combination'] = df.apply(
    lambda row: tuple(sorted([row['color_primary'], row['color_secondary']]))
    if pd.notnull(row['color_secondary']) else None,
    axis=1
)

# Identify dominant two-color combinations
percentage_limit_combinations = 0.05
combination_counts = df['sorted_combination'].dropna().value_counts()
total_combinations = combination_counts.sum()
dominant_combinations = combination_counts[combination_counts / total_combinations > percentage_limit_color].index

df['combination_group'] = df['sorted_combination'].apply(
    lambda x: x if x in dominant_combinations else 'other_multiple_color'
)

# Final color group assignment
df['color_group'] = df.apply(
    lambda row: row['single_color_group']
    if pd.isnull(row['sorted_combination'])
    else row['combination_group'],
    axis=1
)

# Convert tuples in color group to readable strings
df['color_group'] = df['color_group'].apply(
    lambda x: ' & '.join(x) if isinstance(x, tuple) else x
)

##########################
# in sex_upon_intake column, just add if is male or not to avoid having unknown values
df['is_male'] = df['sex_upon_intake'].apply(lambda x: 1 if x =='male' else 0) 
# condition at intake: grouping medical (med attn and medical), normal, sick_injured and rest (other)
df['intake_condition_group'] = df['intake_condition'].apply(
    lambda x: 'normal' if 'normal' in x 
    else 'sick_injured' if 'sick' in x 
    else 'sick_injured' if 'injured' in x 
    else 'medical' if 'medical' in x 
    else 'medical' if 'med attn' in x  
    else 'nursing' if 'nursing' in x 
    else 'nursing' if 'neonatal' in x 
    else 'other')

# group intake type: stray, owner surrended and Other
df['intake_type_group'] = df['intake_type'].apply(
    lambda x: 'stray' if 'stray' in x 
    else 'owner surrender' if 'owner surrender' in x 
    else 'other'
)

####################
# is dog
df['is_dog'] = df['animal_type'].apply(lambda x: 1 if x =='dog' else 0)
######################
# day of intake mothn and day of week
df['day_of_week_in'] = df['datetime_intake'].dt.day_of_week
df['month_in'] = df['datetime_intake'].dt.month

#####################
# Function to calculate the overlap count for a given row, considering animal_type
def count_overlapping_by_type(row, df):
    overlapping = df[
        (df['datetime_intake'] <= row['datetime_outcome']) &  # Shelter intake is before or during this outcome
        (df['datetime_outcome'] >= row['datetime_intake']) &  # Shelter outcome is after or during this intake
       ##(df['animal_type'] == row['animal_type']) &           # Same animal type
       ## (df['breed_group'] == row['breed_group']) &        # Same day of the week
        (df.index != row.name)  # Exclude the current record itself
    ]
    return len(overlapping) 
df['animals_in_shelter'] = df.apply(lambda row: count_overlapping_by_type(row, df), axis=1)

# Drop unused columns
cols_to_drop = ['name', 'age_upon_intake', 'breed_group1', 'breed_group2', 'breed', 'breed_tokenized', 'color', 'color_primary', 'color_secondary', 'single_color_group', 'sorted_combination', 'combination_group']
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

#####################
#####################

# Splitting the dataset
df.reset_index(drop=True, inplace=True)
df_test = df.sort_values('datetime_intake').tail(int(len(df) * 0.2))
print('Test data from:', df_test['datetime_intake'].min(), 'to:', df_test['datetime_intake'].max())

df_full_train = df.drop(df_test.index).reset_index(drop=True)
df_train, df_val = train_test_split(df_full_train, test_size=0.20, random_state=1)

y_full_train = df_full_train['days_in_shelter'].values
y_train = df_train['days_in_shelter'].values
y_val = df_val['days_in_shelter'].values
y_test = df_test['days_in_shelter'].values

del df_full_train['days_in_shelter']
del df_train['days_in_shelter']
del df_val['days_in_shelter']
del df_test['days_in_shelter']

# Define your feature categories
numerical = [
    'age_in_months', 
    'month_in', 
    'animals_in_shelter',
    'day_of_week_in',
    'is_dog',
    'is_mix_breed', 
    'is_miniature', 
    'is_domestic',
    'is_castrated'] 
  
categorical = [    
    'intake_type_group',
    'intake_condition_group',  
    'is_male', 
    'hair_type',
    'breed_group',     
    'color_group'
]

# Training function
def train(df_train, y_train, params):
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    
    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    
    model = xgb.train(params, dtrain, num_boost_round=168, evals=[(dtrain, 'train'), (dval, 'val')], verbose_eval=5, early_stopping_rounds=5)
    
    return dv, model

# Prediction function
def predict(df, dv, model):
    data_dict = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(data_dict)
    features = list(dv.get_feature_names_out())  # Ensure it's a list
    dmatrix = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dmatrix)
    return y_pred

# XGBoost parameters
xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'gamma': gamma,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'min_child_weight': min_child_weight,
    'nthread': nthread,
    'seed': seed,
    'verbosity': verbosity,
    'eval_metric': eval_metric
    
}
# Training the final model
print('Training the final model...')
dv, model = train(df_train, y_train, xgb_params)
y_pred = predict(df_val, dv, model)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse:.3f}')

# Training the final model
print('Training the final model...')
dv, model = train(df_full_train, y_full_train, xgb_params)

# Test set evaluation
y_pred = predict(df_test, dv, model)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test RMSE: {rmse:.3f}')

# Save the final model and DictVectorizer
output_file = f'xgboost_model{eta}_{max_depth}_{gamma}_{subsample}_{colsample_bytree}_{min_child_weight}.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model and DictVectorizer are saved to {output_file}')
