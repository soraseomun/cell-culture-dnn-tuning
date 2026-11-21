import src.functions as sf

# Combine the 3 grid-search result files
filenames = ["grid_search/table_search_set1.csv",
             "grid_search/table_search_set2.csv",
             "grid_search/table_search_set3.csv"]

df_results = sf.combine_tables(filenames)

# Check the number of result rows and save them
print(f"Total number of rows: {len(df_results)}")
print()

sf.save_table(df_results, "grid_search/table_search_combined.csv")

# Rank the hyperparameter combinations and display the best 5
df_ranking = sf.rank_hyperparameters(df_results)

print("Top 5 configurations:")
print(df_ranking.iloc[:5])
print()

sf.save_table(df_ranking, "grid_search/table_search_ranking.csv")

# Display the best configuration in its entirety
print("Best configuration:")
print(df_ranking.iloc[0])
print()
