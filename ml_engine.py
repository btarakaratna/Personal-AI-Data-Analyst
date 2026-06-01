"""
ml_engine.py — Personal AI Data Analyst
Full ML pipeline: preprocessing, training, evaluation,
feature importance, and predictions.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any

warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model      import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree              import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster           import KMeans, DBSCAN
from sklearn.decomposition     import PCA
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
    silhouette_score,
)
from sklearn.impute            import SimpleImputer

# ─── CONSTANTS ────────────────────────────────────────────

PLOTLY_DARK_LAYOUT = dict(
    template        = "plotly_dark",
    paper_bgcolor   = "#0A1628",
    plot_bgcolor    = "#0F1F35",
    font            = dict(family="Poppins, sans-serif", color="#E8F4F8"),
    title_font      = dict(size=16, color="#00F5FF"),
    margin          = dict(l=20, r=20, t=50, b=20),
)

COLOR_PALETTE = ["#00F5FF", "#7B61FF", "#00FFA3", "#FF6B35", "#FF2D78",
                 "#FFD700", "#A78BFA", "#34D399", "#F472B6", "#60A5FA"]

CLASSIFIERS = {
    "🌲 Random Forest":        "random_forest_clf",
    "📊 Logistic Regression":  "logistic_regression",
    "🌳 Decision Tree":        "decision_tree_clf",
    "🚀 Gradient Boosting":    "gradient_boosting",

    "🧠 KNN Classifier":       "knn_classifier",
    "⚡ Support Vector Machine": "svm_classifier",
    "📨 Naive Bayes":          "naive_bayes",
    "🌟 AdaBoost":             "adaboost_classifier",
    "🌲 Extra Trees":          "extra_trees_classifier",
}

REGRESSORS = {
    "📈 Linear Regression":   "linear_regression",
    "🌲 Random Forest":        "random_forest_reg",
    "🌳 Decision Tree":        "decision_tree_reg",
    "🏔️ Ridge Regression":     "ridge_regression",
    "🔱 Lasso Regression":     "lasso_regression",
}

CLUSTERERS = {
    "🔵 KMeans Clustering":   "kmeans",
}


# ─── PREPROCESSING ────────────────────────────────────────

class DataPreprocessor:
    """Automatic preprocessing pipeline for ML tasks."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler        = StandardScaler()
        self.imputer       = SimpleImputer(strategy="median")
        self.feature_names_: list = []
        self.is_fitted_    = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features and target for ML.
        Returns (X, y, feature_names)
        """
        df = df.copy()

        # Select features
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        # Drop cols with >50% missing
        valid_feat = [c for c in feature_cols
                      if df[c].isnull().mean() < 0.5]
        df = df[valid_feat + [target_col]].copy()

        # Encode categorical features
        cat_cols = df[valid_feat].select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Encode target if object
        if df[target_col].dtype == object:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].fillna("Unknown").astype(str))
            self.label_encoders[f"__target_{target_col}"] = le

        # Fill remaining numeric NaN
        X = df[valid_feat].values
        y = df[target_col].values

        self.imputer.fit(X)
        X = self.imputer.transform(X)

        if scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        self.feature_names_ = valid_feat
        self.is_fitted_      = True
        return X, y, valid_feat

    def decode_target(self, target_col: str, y: np.ndarray) -> np.ndarray:
        """Reverse-encode target labels."""
        key = f"__target_{target_col}"
        if key in self.label_encoders:
            return self.label_encoders[key].inverse_transform(y.astype(int))
        return y


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Auto-detect 'classification' or 'regression' based on target column."""
    target = df[target_col].dropna()
    if pd.api.types.is_numeric_dtype(target):
        n_unique = target.nunique()
        if n_unique <= 10:
            return "classification"
        return "regression"
    return "classification"


# ─── MODEL FACTORY ────────────────────────────────────────

def get_model(model_key: str, n_estimators: int = 100, max_depth: int = None, n_clusters: int = 3):
    """Instantiate and return a scikit-learn model by key."""
    models = {
        "random_forest_clf":  RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=42, n_jobs=-1),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree_clf":   DecisionTreeClassifier(max_depth=max_depth or 6, random_state=42),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=n_estimators, random_state=42),
        "knn_classifier": KNeighborsClassifier(
            n_neighbors=5
        ),

        "svm_classifier": SVC(
            probability=True,
            random_state=42
        ),

        "naive_bayes": GaussianNB(),

        "adaboost_classifier": AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=42
        ),

        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        ),
        "linear_regression":   LinearRegression(),
        "random_forest_reg":   RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=42, n_jobs=-1),
        "decision_tree_reg":   DecisionTreeRegressor(max_depth=max_depth or 6, random_state=42),
        "ridge_regression":    Ridge(alpha=1.0),
        "lasso_regression":    Lasso(alpha=0.1, max_iter=5000),
        "kmeans":              KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    }
    return models.get(model_key)


# ─── CLASSIFICATION PIPELINE ──────────────────────────────

def run_classification(
    df: pd.DataFrame,
    target_col: str,
    model_key: str = "random_forest_clf",
    test_size: float = 0.2,
    feature_cols: Optional[list] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> dict:
    """
    Full classification pipeline.
    Returns comprehensive result dict.
    """
    result = {
        "task": "classification",
        "model_key": model_key,
        "target_col": target_col,
        "success": False,
        "model": None,
        "metrics": {},
        "feature_importance": None,
        "confusion_matrix": None,
        "predictions_df": None,
        "fig_importance": None,
        "fig_confusion": None,
        "fig_metrics": None,
        "class_report": "",
        "error": None,
    }

    try:
        preprocessor = DataPreprocessor()
        X, y, feat_names = preprocessor.fit_transform(
            df, target_col, feature_cols=feature_cols
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        model = get_model(model_key, n_estimators=n_estimators, max_depth=max_depth)
        if model is None:
            result["error"] = f"Model '{model_key}' not found."
            return result

        model.fit(X_train, y_train)
        result["model"] = model
        y_pred = model.predict(X_test)

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        try:
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        except Exception:
            f1 = prec = rec = 0.0

        result["metrics"] = {
            "Accuracy":  round(acc * 100, 2),
            "F1 Score":  round(f1 * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall":    round(rec * 100, 2),
            "Train Size": len(X_train),
            "Test Size":  len(X_test),
        }

        # Classification report
        try:
            result["class_report"] = classification_report(y_test, y_pred)
        except Exception:
            pass

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        result["confusion_matrix"] = cm
        labels = np.unique(np.concatenate([y_test, y_pred])).astype(str)
        # Decode labels if encoded
        decoded_labels = preprocessor.decode_target(target_col, np.unique(np.concatenate([y_test, y_pred]))).astype(str)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=decoded_labels, y=decoded_labels,
            colorscale=[[0, "#0F1F35"], [0.5, "#7B61FF"], [1, "#00F5FF"]],
            showscale=True,
            text=cm.astype(str), texttemplate="%{text}",
            textfont={"size": 14},
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            **PLOTLY_DARK_LAYOUT,
        )
        result["fig_confusion"] = fig_cm

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature":    feat_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=True).tail(15)

            fig_imp = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale=["#7B61FF", "#00F5FF"],
            )
            fig_imp.update_layout(**PLOTLY_DARK_LAYOUT)
            result["feature_importance"] = feat_df.sort_values("Importance", ascending=False)
            result["fig_importance"]     = fig_imp

        # Predictions table
        y_pred_labels = preprocessor.decode_target(target_col, y_pred)
        y_test_labels = preprocessor.decode_target(target_col, y_test)
        pred_df = pd.DataFrame({
            "Actual":    y_test_labels,
            "Predicted": y_pred_labels,
            "Correct":   (y_test == y_pred),
        })
        result["predictions_df"] = pred_df.reset_index(drop=True)

        # Metrics radar chart
        metric_names  = ["Accuracy", "F1 Score", "Precision", "Recall"]
        metric_values = [result["metrics"][m] for m in metric_names]
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatterpolar(
            r=metric_values + [metric_values[0]],
            theta=metric_names + [metric_names[0]],
            fill="toself",
            fillcolor="rgba(0,245,255,0.1)",
            line=dict(color="#00F5FF", width=2),
            marker=dict(color="#7B61FF", size=8),
            name="Model Performance",
        ))
        fig_metrics.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="#0F1F35",
            ),
            title="Performance Metrics Radar",
            showlegend=False,
            **PLOTLY_DARK_LAYOUT,
        )
        result["fig_metrics"] = fig_metrics
        result["success"]     = True

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-300:]}"

    return result


# ─── REGRESSION PIPELINE ──────────────────────────────────

def run_regression(
    df: pd.DataFrame,
    target_col: str,
    model_key: str = "random_forest_reg",
    test_size: float = 0.2,
    feature_cols: Optional[list] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> dict:
    """Full regression pipeline."""
    result = {
        "task": "regression",
        "model_key": model_key,
        "target_col": target_col,
        "success": False,
        "model": None,
        "metrics": {},
        "feature_importance": None,
        "predictions_df": None,
        "fig_importance": None,
        "fig_actual_vs_pred": None,
        "fig_residuals": None,
        "error": None,
    }

    try:
        preprocessor = DataPreprocessor()
        X, y, feat_names = preprocessor.fit_transform(
            df, target_col, feature_cols=feature_cols
        )
        y = y.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = get_model(model_key, n_estimators=n_estimators, max_depth=max_depth)
        if model is None:
            result["error"] = f"Model '{model_key}' not found."
            return result

        model.fit(X_train, y_train)
        result["model"] = model
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        result["metrics"] = {
            "R² Score":  round(float(r2), 4),
            "RMSE":      round(float(rmse), 4),
            "MAE":       round(float(mae), 4),
            "Train Size": len(X_train),
            "Test Size":  len(X_test),
        }

        # Predictions df
        pred_df = pd.DataFrame({
            "Actual":    y_test.round(2),
            "Predicted": y_pred.round(2),
            "Error":     (y_test - y_pred).round(2),
            "Abs Error": np.abs(y_test - y_pred).round(2),
        }).reset_index(drop=True)
        result["predictions_df"] = pred_df

        # Actual vs Predicted
        fig_ap = px.scatter(
            pred_df, x="Actual", y="Predicted",
            title="Actual vs Predicted",
            color="Abs Error",
            color_continuous_scale=["#00FFA3", "#00F5FF", "#7B61FF", "#FF2D78"],
            opacity=0.7,
        )
        # Perfect prediction line
        min_val = float(min(y_test.min(), y_pred.min()))
        max_val = float(max(y_test.max(), y_pred.max()))
        fig_ap.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Fit",
            line=dict(color="#00F5FF", width=2, dash="dash"),
        ))
        fig_ap.update_layout(**PLOTLY_DARK_LAYOUT)
        result["fig_actual_vs_pred"] = fig_ap

        # Residuals
        residuals = y_test - y_pred
        fig_res = px.histogram(
            x=residuals, nbins=30,
            title="Residuals Distribution",
            labels={"x": "Residual", "y": "Count"},
            color_discrete_sequence=["#7B61FF"],
        )
        fig_res.update_layout(**PLOTLY_DARK_LAYOUT)
        result["fig_residuals"] = fig_res

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature":    feat_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=True).tail(15)

            fig_imp = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale=["#7B61FF", "#00F5FF"],
            )
            fig_imp.update_layout(**PLOTLY_DARK_LAYOUT)
            result["feature_importance"] = feat_df.sort_values("Importance", ascending=False)
            result["fig_importance"]     = fig_imp

        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_) if model.coef_.ndim == 1 else np.abs(model.coef_[0])
            feat_df = pd.DataFrame({
                "Feature":    feat_names[:len(coefs)],
                "Importance": coefs,
            }).sort_values("Importance", ascending=True).tail(15)
            fig_imp = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                title="Coefficient Magnitudes",
                color="Importance",
                color_continuous_scale=["#7B61FF", "#00F5FF"],
            )
            fig_imp.update_layout(**PLOTLY_DARK_LAYOUT)
            result["feature_importance"] = feat_df.sort_values("Importance", ascending=False)
            result["fig_importance"]     = fig_imp

        result["success"] = True

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-300:]}"

    return result


# ─── CLUSTERING PIPELINE ──────────────────────────────────

def run_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    n_clusters: int = 3,
    model_key: str = "kmeans",
) -> dict:
    """Full KMeans clustering pipeline."""
    result = {
        "task": "clustering",
        "model_key": model_key,
        "n_clusters": n_clusters,
        "success": False,
        "metrics": {},
        "cluster_df": None,
        "fig_clusters": None,
        "fig_elbow": None,
        "fig_distribution": None,
        "error": None,
    }

    try:
        preprocessor = DataPreprocessor()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if feature_cols is None:
            feature_cols = numeric_cols[:10]  # max 10 features

        if len(feature_cols) < 2:
            result["error"] = "Need at least 2 numeric features for clustering."
            return result

        X, _, feat_names = preprocessor.fit_transform(
            df.assign(__dummy__=0), "__dummy__",
            feature_cols=feature_cols
        )

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        # Silhouette score
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = 0.0

        result["metrics"] = {
            "Silhouette Score": round(float(sil), 4),
            "Inertia":          round(float(model.inertia_), 2),
            "N Clusters":       n_clusters,
            "N Points":         len(X),
        }

        # Add clusters to original dataframe
        cluster_df = df[feature_cols].copy()
        cluster_df["Cluster"] = [f"Cluster {l+1}" for l in labels]
        result["cluster_df"] = cluster_df

        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        variance = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "PC1":     X_pca[:, 0],
            "PC2":     X_pca[:, 1],
            "Cluster": [f"Cluster {l+1}" for l in labels],
        })

        fig_clusters = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title=f"KMeans Clustering (PCA) — {n_clusters} Clusters",
            color_discrete_sequence=COLOR_PALETTE,
            opacity=0.8,
        )
        fig_clusters.update_traces(marker=dict(size=8))
        fig_clusters.update_layout(
            xaxis_title=f"PC1 ({variance[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({variance[1]*100:.1f}% variance)",
            **PLOTLY_DARK_LAYOUT,
        )
        result["fig_clusters"] = fig_clusters

        # Elbow curve
        inertias = []
        k_range  = range(1, min(10, len(X)) + 1)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X)
            inertias.append(km.inertia_)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers",
            line=dict(color="#00F5FF", width=2),
            marker=dict(color="#7B61FF", size=8),
            name="Inertia",
        ))
        fig_elbow.add_vline(
            x=n_clusters, line_dash="dash",
            line_color="#FF6B35", annotation_text=f"k={n_clusters}",
        )
        fig_elbow.update_layout(
            title="Elbow Curve — Optimal K",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            **PLOTLY_DARK_LAYOUT,
        )
        result["fig_elbow"] = fig_elbow

        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        cluster_counts["Cluster"] = cluster_counts["Cluster"].apply(lambda x: f"Cluster {x+1}")

        fig_dist = px.bar(
            cluster_counts.sort_values("Cluster"),
            x="Cluster", y="Count",
            title="Cluster Size Distribution",
            color="Cluster",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_dist.update_layout(**PLOTLY_DARK_LAYOUT)
        result["fig_distribution"] = fig_dist

        result["success"] = True

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-300:]}"

    return result


# ─── AUTO ML ──────────────────────────────────────────────

def auto_ml(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> dict:
    """
    Automatically detect task type and run the best default model.
    """
    task = detect_task_type(df, target_col)
    if task == "classification":
        return run_classification(
            df, target_col,
            model_key="random_forest_clf",
            test_size=test_size,
        )
    else:
        return run_regression(
            df, target_col,
            model_key="random_forest_reg",
            test_size=test_size,
        )
