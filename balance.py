import pandas as pd

#load csv
df = pd.read_csv("result/result.csv")

# Define the number of samples to select for each label
desired_samples_per_label = 2500

# Create an empty DataFrame to store the balanced dataset
balanced_df = pd.DataFrame()

# Iterate over each unique label in the DataFrame
for label in df['label'].unique():
    # Filter the DataFrame to get samples for the current label
    label_df = df[df['label'] == label]
    
    # Sample the desired number of samples for the current label
    sampled_df = label_df.sample(min(desired_samples_per_label, len(label_df)), random_state=42)
    
    # Append the sampled DataFrame to the balanced dataset
    balanced_df = balanced_df._append(sampled_df, ignore_index=True)

# Display the balanced DataFrame
print(balanced_df.head())

# Save the balanced DataFrame to a new CSV file
balanced_df.to_csv('result/result_balanced.csv', index=False)

print("Balanced dataset saved successfully.")