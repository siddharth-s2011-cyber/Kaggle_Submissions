import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train:", train.shape)
print("Test :", test.shape)


# =========================
# 2. BASIC INSPECTION
# =========================

print("\n--- HEAD ---")
print(train.head())

print("\n--- INFO ---")
print(train.info())

print("\n--- DESCRIBE ---")
print(train.describe(include="all").T)


# =========================
# 3. COLUMN TYPES
# =========================

numeric_cols = train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical:", numeric_cols)
print("Categorical:", categorical_cols)


# =========================
# 4. DUPLICATES
# =========================

print("\nTrain duplicates:",
      train.duplicated().sum())

print("Test duplicates:",
      test.duplicated().sum())


# =========================
# 5. UNIQUE VALUES
# =========================

print("\n--- UNIQUE VALUES ---")
print(train.nunique().sort_values())


# =========================
# 6. MISSING VALUES
# =========================

missing = pd.DataFrame({
    "Train Missing": train.isna().sum(),
    "Train %": train.isna().mean() * 100,
    "Test Missing": test.isna().sum(),
    "Test %": test.isna().mean() * 100
})

print("\n--- MISSING VALUES ---")
print(
    missing.sort_values("Train %", ascending=False)
)


# =========================
# 7. CATEGORICAL VALUES
# =========================

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(
        train[col].value_counts(dropna=False)
    )


# =========================
# 8. TARGET
# =========================

target = "Survived"
print("\n--- TARGET DISTRIBUTION ---")
print(train[target].value_counts())
print(train[target].value_counts(normalize=True) * 100
)
sns.countplot(
    data=train,
    x=target
)
plt.title("Target Distribution")
plt.show()
# =========================
# 9. NUMERICAL DISTRIBUTIONS
# =========================

for col in numeric_cols:
    if col == target:
        continue
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=train,
        x=col,
        kde=True
    )
    plt.title(f"{col} Distribution")
    plt.show()
# =========================
# 10. CATEGORICAL DISTRIBUTIONS
# =========================
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=train,
        x=col
    )
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.show()
# =========================
# 11. NUMERICAL → TARGET
# =========================
for col in numeric_cols:

    if col == target:
        continue

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=train,
        x=target,
        y=col
    )

    plt.title(f"{col} vs {target}")
    plt.show()


# =========================
# 12. CATEGORICAL → TARGET
# =========================

for col in categorical_cols:

    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=train,
        x=col,
        hue=target
    )

    plt.title(f"{col} vs {target}")
    plt.xticks(rotation=45)
    plt.show()


# =========================
# 13. CORRELATION
# =========================

corr = train[
    numeric_cols
].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()
