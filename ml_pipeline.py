import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

# Optional heavy dependencies — imported lazily inside methods to avoid startup crashes
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

class MLEngine:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.models = {}

    def load_data(self, path):
        print(f"Loading data from {path}...")
        return pd.read_parquet(path)

    def clean_data(self, df):
        print("Cleaning data...")
        df = df.drop_duplicates()
        df = df.fillna(0) 
        return df

    def feature_engineering(self, df):
        print("Engineering features...")
        df = df.copy()
        df['price_ratio'] = df.get('price_seen_usd', df.get('price_seen', 0)) / (df.get('base_price_usd', df.get('base_price', 1)) + 1e-5)
        df['engagement_index'] = df.get('session_intensity', 0) * df.get('click_frequency', 0)
        return df

    def create_labels(self, df):
        print("Creating labels...")
        if 'checkout_status' in df.columns:
            df['label_pricing'] = (df['checkout_status'] == 'completed').astype(int)
        if 'clicked_recom' in df.columns:
            df['label_recom'] = (df['clicked_recom'] == True).astype(int)
        return df

    def data_split(self, df, label_col):
        drop_cols = ['label_pricing', 'label_recom', 'user_id', 'timestamp', 'checkout_status', 'clicked_recom']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Only keep numeric/encoded columns generally or we can rely on scale_and_encode
        
        # Ensure label column exists
        if label_col not in df.columns:
            # Fallback if label is missing
            print(f"Warning: {label_col} not found, generating dummy labels for testing")
            y = np.random.randint(0, 2, size=len(df))
        else:
            y = df[label_col]
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def scale_and_encode(self, X_train, X_val, cat_cols, num_cols):
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        # Filter existing categorical columns
        existing_cat_cols = [c for c in cat_cols if c in X_train.columns]
        for col in existing_cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_val[col] = X_val[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )
            self.encoders[col] = le
            
        # Ensure numeric types
        X_train = X_train.select_dtypes(include=[np.number])
        X_val = X_val[X_train.columns].select_dtypes(include=[np.number])

        existing_num_cols = [c for c in num_cols if c in X_train.columns]
        
        if existing_num_cols:
            scaler = StandardScaler()
            X_train[existing_num_cols] = scaler.fit_transform(X_train[existing_num_cols])
            X_val[existing_num_cols] = scaler.transform(X_val[existing_num_cols])
            self.scalers['standard'] = scaler
            
        return X_train, X_val

    def train_pricing_model(self, X_train, y_train):
        print("Training Pricing Model...")
        # Ensure X_train is fully numeric
        if _XGB_AVAILABLE:
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1,
                                      verbosity=0, eval_metric="logloss")
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            print("  [WARN] XGBoost not installed — falling back to sklearn GBM")
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                               random_state=42)
        model.fit(X_train, y_train)
        self.models['pricing'] = model
        return model

    def train_recom_model(self, X_train, y_train):
        print("Training Recommendation Model...")
        if _LGB_AVAILABLE:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       n_jobs=-1, verbose=-1)
        else:
            from sklearn.linear_model import LogisticRegression
            print("  [WARN] LightGBM not installed — falling back to LogisticRegression")
            model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        self.models['recom'] = model
        return model

    def evaluate(self, model, X_val, y_val, name="Model"):
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        print(f"\n{'='*40}\n  {name} Evaluation\n{'='*40}")
        print(f"AUC-ROC: {roc_auc_score(y_val, probs):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, preds, zero_division=0))

    def save_pipeline(self, filepath="trained_models/apex_pipeline_v3.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders
        }, filepath)
        print(f"Pipeline successfully saved to {filepath}")
        
    @classmethod
    def load_pipeline(cls, filepath="trained_models/apex_pipeline_v3.pkl"):
        if not os.path.exists(filepath):
            return None
        data = joblib.load(filepath)
        engine = cls()
        engine.models = data.get('models', {})
        engine.scalers = data.get('scalers', {})
        engine.encoders = data.get('encoders', {})
        return engine
        
    def predict_pricing(self, df):
        # Applies minimal feature engineering and scaling, assumes cat_cols are encoded if provided
        df = self.feature_engineering(df)
        df_copy = df.copy()
        
        for col, le in self.encoders.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )
                
        df_numeric = df_copy.select_dtypes(include=[np.number])
        if 'standard' in self.scalers:
            try:
                scaler = self.scalers['standard']
                cols_to_scale = [c for c in scaler.feature_names_in_ if c in df_numeric.columns]
                if cols_to_scale:
                    df_numeric[cols_to_scale] = scaler.transform(df_numeric[cols_to_scale])
            except Exception:
                pass
                
        # Get features model was trained on
        model = self.models.get('pricing')
        if not model:
            return np.array([0.5])  # neutral fallback instead of None

        try:
            features = model.feature_names_in_
        except AttributeError:
            # sklearn fallback models may not have feature_names_in_
            features = df_numeric.columns.tolist()

        for f in features:
            if f not in df_numeric.columns:
                df_numeric[f] = 0

        X = df_numeric[list(features)]
        return model.predict_proba(X)[:, 1]

    def predict_recom(self, df):
        df = self.feature_engineering(df)
        df_copy = df.copy()
        
        for col, le in self.encoders.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )
                
        df_numeric = df_copy.select_dtypes(include=[np.number])

        model = self.models.get('recom')
        if not model:
            return np.array([0.5])  # neutral fallback instead of None

        try:
            features = model.feature_names_in_
        except AttributeError:
            features = df_numeric.columns.tolist()

        for f in features:
            if f not in df_numeric.columns:
                df_numeric[f] = 0

        X = df_numeric[list(features)]
        return model.predict_proba(X)[:, 1]

if __name__ == "__main__":
    pipeline = MLEngine()
    
    cat_cols = ['category', 'device_type']
    num_cols = ['price_seen_usd', 'base_price_usd', 'price_ratio', 'engagement_index']
    
    # --- PRICING MODEL PIPELINE ---
    if os.path.exists("prepared_data/pricing_features_train.parquet"):
        df_pricing = pipeline.load_data("prepared_data/pricing_features_train.parquet")
        df_pricing = pipeline.clean_data(df_pricing)
        df_pricing = pipeline.feature_engineering(df_pricing)
        
        # Use existing label or create dummy
        label_p = 'label_conversion' if 'label_conversion' in df_pricing.columns else 'label_pricing'
        df_pricing = pipeline.create_labels(df_pricing)
        if label_p not in df_pricing.columns:
             df_pricing[label_p] = np.random.randint(0, 2, size=len(df_pricing))
        
        X_train_p, X_val_p, y_train_p, y_val_p = pipeline.data_split(df_pricing, label_p)
        X_train_p, X_val_p = pipeline.scale_and_encode(X_train_p, X_val_p, cat_cols, num_cols)
        
        model_p = pipeline.train_pricing_model(X_train_p, y_train_p)
        pipeline.evaluate(model_p, X_val_p, y_val_p, "Pricing Model")
    
    # --- RECOMMENDATION MODEL PIPELINE ---
    if os.path.exists("prepared_data/recommendation_features_train.parquet"):
        df_recom = pipeline.load_data("prepared_data/recommendation_features_train.parquet")
        df_recom = pipeline.clean_data(df_recom)
        df_recom = pipeline.feature_engineering(df_recom)
        
        label_r = 'label_clicked' if 'label_clicked' in df_recom.columns else 'label_recom'
        df_recom = pipeline.create_labels(df_recom)
        if label_r not in df_recom.columns:
             df_recom[label_r] = np.random.randint(0, 2, size=len(df_recom))
             
        X_train_r, X_val_r, y_train_r, y_val_r = pipeline.data_split(df_recom, label_r)
        X_train_r, X_val_r = pipeline.scale_and_encode(X_train_r, X_val_r, cat_cols, num_cols)
        
        model_r = pipeline.train_recom_model(X_train_r, y_train_r)
        pipeline.evaluate(model_r, X_val_r, y_val_r, "Recommendation Model")
    
    # Save the unified object
    pipeline.save_pipeline("trained_models/apex_pipeline_v3.pkl")

