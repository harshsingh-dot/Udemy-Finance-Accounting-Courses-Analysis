import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# === Step 1: Set Base Directory and Data Path ===
# This ensures paths work regardless of where you run the script from
base_dir = Path(r"E:\Udemy_Course_Project")
data_path = base_dir / "data" / "udemy_output_All_Finance__Accounting_p1_p626.csv"

# Ensure the file exists before proceeding
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at: {data_path}")

# === Step 2: Load Data ===
df = pd.read_csv(data_path)

# === Step 3: Clean & Preprocess ===
df.columns = df.columns.str.strip()
df['created'] = pd.to_datetime(df['created'], errors='coerce')
df['published_time'] = pd.to_datetime(df['published_time'], errors='coerce')

# Convert INR to USD
inr_to_usd = 1 / 82
df['discount_price__amount'] = df['discount_price__amount'] * inr_to_usd
df['price_detail__amount'] = df['price_detail__amount'] * inr_to_usd

# Calculate Discount %
df['Discount_Percentage'] = ((df['price_detail__amount'] - df['discount_price__amount']) / df['price_detail__amount']) * 100

# Impute missing numeric values
num_cols = ['discount_price__amount', 'price_detail__amount', 'Discount_Percentage']
df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

# Clean course titles
df['title'] = (
    df['title']
    .str.lower()
    .str.strip()
    .replace(r'\s+', ' ', regex=True)
    .str.replace(r'[^\w\s]', '', regex=True)
)

# === Step 4: Feature Engineering ===
def categorize(title):
    if 'excel' in title or 'spreadsheet' in title:
        return 'Spreadsheet'
    elif 'sql' in title or 'database' in title:
        return 'Database'
    elif 'tableau' in title or 'power bi' in title:
        return 'Visualization'
    elif 'finance' in title or 'accounting' in title:
        return 'Finance'
    elif 'pmp' in title or 'project' in title:
        return 'Project Mgmt'
    elif 'data' in title or 'machine' in title:
        return 'Data Science'
    elif 'business' in title:
        return 'Business'
    else:
        return 'Other'

df['category'] = df['title'].apply(categorize)

# === Step 5: Model Preparation ===
features = ['rating', 'num_reviews', 'num_published_lectures',
            'discount_price__amount', 'price_detail__amount', 'Discount_Percentage']
target = 'num_subscribers'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Train Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 7: Evaluation ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# === Step 8: Cross-Validation ===
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n✅ Cross-Validation R² (mean): {cv_scores.mean():.4f}")
print(f"✅ Cross-Validation R² (std): {cv_scores.std():.4f}")

# === Step 9: Visualize ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Subscribers")
plt.ylabel("Predicted Subscribers")
plt.title("Actual vs Predicted Number of Subscribers")
plt.tight_layout()

# Save plot in project folder
output_plot = base_dir / "predicted_vs_actual.png"
plt.savefig(output_plot)
plt.show()

print(f"\n📁 Plot saved at: {output_plot}")
