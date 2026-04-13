import pandas as pd

# Load RNN/LSTM/GRU results
df1 = pd.read_csv("model_results.csv")

# Load BERT results
df2 = pd.read_csv("bert_result.csv")

# Combine
final_df = pd.concat([df1, df2], ignore_index=True)

# Save
final_df.to_csv("final_model_comparison.csv", index=False)

print("✅ Final comparison table created!")
print(final_df)