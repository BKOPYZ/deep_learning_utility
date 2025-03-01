import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def identify_missing_values(df):
    pd.set_option("display.max_rows", None)
    missing_values = df.isnull().sum()
    pd.reset_option("display.max_rows")
    return missing_values


def handle_missing_values(df, strategy="mean", fill_value=None):
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed


def remove_duplicates(df):
    return df.drop_duplicates()


def identify_outliers(df, contamination=0.01, seed=42):
    iso = IsolationForest(contamination=contamination, random_state=seed)
    outliers = iso.fit_predict(df)
    return df[outliers == 1], df[outliers == -1]


def convert_data_types(df, column_types):
    return df.astype(column_types)


def normalize_scale(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled


def encode_categorical(df, encoding_type="onehot"):
    if encoding_type == "onehot":
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df)
        df_encoded = pd.DataFrame(encoded, columns=df.columns)
        return df_encoded
    elif encoding_type == "label":
        encoder = LabelEncoder()
        for col in df.columns:
            df[col] = encoder.fit_transform(df[col])
        return df


def feature_engineering(df, new_features: dict[str, callable]):
    for feature_name, feature_func in new_features.items():
        df[feature_name] = df.apply(feature_func, axis=1)
    return df


def clean_text_data(df, text_columns):

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def clean_text(text):
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(
            [stemmer.stem(word) for word in text.split() if word not in stop_words]
        )
        return text

    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    return df


def extract_date_features(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day
        df[col + "_hour"] = df[col].dt.hour
    return df


def address_imbalanced_data(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def ensure_consistent_formatting(df, string_columns, date_columns):
    for col in string_columns:
        df[col] = df[col].str.strip().str.lower()
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df


def integrate_data(df_list, on_columns):
    return pd.concat(df_list, join="inner", on=on_columns)


def remove_irrelevant_data(df, columns_to_remove):
    return df.drop(columns=columns_to_remove)


def validate_data(df, validation_func):
    return validation_func(df)
