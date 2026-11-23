Data Preprocessing & Categorical Encoding Practice

Hands-on practice of the first steps of a machine learning workflow: cleaning data and converting categorical (text) columns into numbers. Built while learning ML from scratch, using the Titanic dataset plus small toy examples.

Status: Work in progress. This repo is part of my learning journey toward an AI/ML internship. No models have been trained yet. I am finishing preprocessing and encoding first.

What I practiced
#	Topic	What I did
1	Loading and inspecting data	read_csv, head(), shape, info(), isnull().sum()
2	Dropping useless columns	Removed PassengerId, Name, Ticket, Cabin
3	Handling missing values	Age filled with median, Embarked filled with mode
4	Manual encoding with map	Sex: male = 1, female = 0
5	Splitting features and target	X (6 features) and y (Survived)
6	Train/test split	train_test_split with test_size=0.25, random_state=42
7	Ordinal Encoding	OrdinalEncoder on an education column (toy data)
8	Label Encoding	LabelEncoder on Yes/No, cities and colors (toy data)
9	One-Hot Encoding	pd.get_dummies (in progress, Titanic Embarked next)
Dataset

Titanic: Machine Learning from Disaster (train.csv) from Kaggle. 891 rows, 12 columns originally.

Target: Survived (0 = did not survive, 1 = survived)
Missing values found: Age (177), Cabin (687), Embarked (2)
Steps and code
1. Load and inspect
python
import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.shape)            # (891, 12)
print(df.isnull().sum())
df.info()
2. Drop columns that don't help prediction
python
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
print(df.shape)            # (891, 8)
Column	Reason
PassengerId	Just a serial number
Name	Every value is unique
Ticket	Mostly unique values
Cabin	About 77% missing
3. Fill missing values
python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
Numeric column uses the median, which is not pulled by very large values.
Text column uses the mode (most frequent value). mode() returns a Series, so [0] picks the first value.
4. Encode Sex (binary column)
python
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
5. Separate features and target
python
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]
print(X.shape, y.shape)    # (891, 6) (891,)

Embarked is left out for now until One-Hot Encoding is applied to it.

6. Train/test split
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# X_train: (668, 6), X_test: (223, 6)

The model learns only from the training set. The test set is kept aside to check how it performs on data it has never seen.

Encoding techniques (toy examples)
Ordinal Encoding: when categories have a real order
python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "name": ["Ali", "Sara", "Ahmed", "Zoya", "Bilal"],
    "education": ["Inter", "BS", "Matric", "MS", "BS"]
})

oe = OrdinalEncoder(categories=[["Matric", "Inter", "BS", "MS"]])
df["education_enc"] = oe.fit_transform(df[["education"]])

Result: Matric = 0, Inter = 1, BS = 2, MS = 3. The categories argument is required. Without it, sklearn sorts alphabetically, which gives a wrong order.

Label Encoding: mostly for the target column
python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["placed_enc"] = le.fit_transform(df["placed"])
print(le.classes_)         # ['No' 'Yes']  ->  No = 0, Yes = 1
Assigns numbers in alphabetical order, starting from 0.
The same value always gets the same number.
Takes a 1D input (df["col"]), while OrdinalEncoder takes 2D (df[["col"]]).
One-Hot Encoding: when categories have no order
python
df = pd.DataFrame({
    "color": ["Red", "Blue", "Green", "Blue", "Yellow"],
    "price": [500, 300, 400, 320, 450]
})

df = pd.get_dummies(df, columns=["color"], dtype=int)

Creates one 0/1 column per category (color_Blue, color_Green, ...), so the model does not assume a fake order between colors.

Which encoding to use
Situation	Encoding
Categories have a real order (Small < Medium < Large)	Ordinal
Target column (Yes/No, Spam/Not Spam)	Label
Categories with no order (city, color, port)	One-Hot
What I learned
A model cannot work with text or empty cells, so data must be cleaned and converted to numbers first.
The right encoding depends on whether the column has an order.
Always fit/transform and save the result back (df = ...), otherwise the dataframe does not change.
Train and test data must be separated before judging a model.
Next steps
[ ] Apply One-Hot Encoding to Titanic Embarked
[ ] Scaling (StandardScaler / MinMaxScaler)
[ ] Learn Logistic Regression and train the first model on Titanic
[ ] Compare other models (Decision Tree, Random Forest)
[ ] Feature engineering (extract title from Name)
[ ] Practice on Students Performance and Telco Churn datasets
Tech stack

Python, pandas, NumPy, scikit-learn, Jupyter Notebook
