import pandas as pd
import numpy as np
data = {
    "Name": [f"Student{i}" for i in range(1, 11)],
    "Subject": np.random.choice(["Math", "Physics", "Chemistry"], 10),
    "Score": np.random.randint(50, 101, 10),
    "Grade": ["" for _ in range(10)]
}
df = pd.DataFrame(data)
def assign_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
df["Grade"] = df["Score"].apply(assign_grade)
print(df.sort_values(by="Score", ascending=False))
print(df.groupby("Subject")["Score"].mean())

def pandas_filter_pass(dataframe):
    return dataframe[dataframe["Grade"].isin(["A", "B"])]
