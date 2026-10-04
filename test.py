import numpy as np
import pandas as pd

data = {
    "Name": ["Ali", "John", "Sarah"],
    "Age": [25, 32, 28],
    "Experience": [2, 8, 5]
}

df = pd.DataFrame(data)

print(df)
print()
print("Average age:", df["Age"].mean())
print("Average experience:", df["Experience"].mean())