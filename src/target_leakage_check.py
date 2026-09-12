import pandas as pd

file_path = "data/milling/milling_cleaned.csv"

data = pd.read_csv(file_path)

print("========== TARGET LEAKAGE CHECK ==========")

print("\nNumberOfCycle:")
print(data["NumberOfCycle"].head(20).to_list())

print("\nCycleToFailure:")
print(data["CycleToFailure"].head(20).to_list())

print("\nLast 20 NumberOfCycle values:")
print(data["NumberOfCycle"].tail(20).to_list())

print("\nLast 20 CycleToFailure values:")
print(data["CycleToFailure"].tail(20).to_list())