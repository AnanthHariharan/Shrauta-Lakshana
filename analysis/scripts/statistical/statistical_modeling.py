"""
Statistical Modeling Module for Vedic Sanskrit Diachronic Analysis

This module implements advanced statistical modeling techniques to:
1. Classify texts by characteristics (prose/poetry, period, recension)
2. Apply linear regression with predictive features
3. Quantify feature importance
4. Build comprehensive models explaining textual differences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class VedicTextMetadata:
    """Comprehensive metadata classification for Vedic texts"""
    
    def __init__(self):
        self.text_metadata = {
            # SAMHITAS - Metrical, Early Period
            'Rigveda': {
                'genre': 'poetry',
                'period': 'early_vedic',
                'chronology': 1500,  # BCE
                'recension': 'shakala',
                'veda_branch': 'rig',
                'text_type': 'samhita',
                'prose_ratio': 0.05,  # Mostly verse
                'liturgical_focus': 'hymnal',
                'geographic_origin': 'northwest',
                'preservation_quality': 'excellent'
            },
            'Yajurveda': {
                'genre': 'mixed',
                'period': 'early_vedic',
                'chronology': 1300,
                'recension': 'maitrayaniya',
                'veda_branch': 'yajur',
                'text_type': 'samhita',
                'prose_ratio': 0.30,  # Mixed verse/prose
                'liturgical_focus': 'sacrificial',
                'geographic_origin': 'northwest',
                'preservation_quality': 'good'
            },
            'Samaveda': {
                'genre': 'poetry',
                'period': 'middle_vedic',
                'chronology': 1200,
                'recension': 'kauthuma',
                'veda_branch': 'sama',
                'text_type': 'samhita',
                'prose_ratio': 0.10,  # Mostly melodic verse
                'liturgical_focus': 'chanting',
                'geographic_origin': 'northwest',
                'preservation_quality': 'good'
            },
            'Atharvaveda (Paippalada)': {
                'genre': 'poetry',
                'period': 'middle_vedic',
                'chronology': 800,
                'recension': 'paippalada',
                'veda_branch': 'atharva',
                'text_type': 'samhita',
                'prose_ratio': 0.20,  # Some prose passages
                'liturgical_focus': 'magical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'fair'
            },
            'Atharvaveda (Saunaka)': {
                'genre': 'poetry',
                'period': 'middle_vedic',
                'chronology': 600,
                'recension': 'shaunaka',
                'veda_branch': 'atharva',
                'text_type': 'samhita',
                'prose_ratio': 0.15,  # Mostly verse
                'liturgical_focus': 'magical',
                'geographic_origin': 'central',
                'preservation_quality': 'good'
            },
            
            # BRAHMANAS - Prose, Late Vedic
            'Kausitaki-Br': {
                'genre': 'prose',
                'period': 'late_vedic',
                'chronology': 900,
                'recension': 'kausitaki',
                'veda_branch': 'rig',
                'text_type': 'brahmana',
                'prose_ratio': 0.95,  # Almost all prose
                'liturgical_focus': 'ritual_explanation',
                'geographic_origin': 'eastern',
                'preservation_quality': 'good'
            },
            'Pancavimsa-Br': {
                'genre': 'prose',
                'period': 'late_vedic',
                'chronology': 850,
                'recension': 'pancavimsa',
                'veda_branch': 'sama',
                'text_type': 'brahmana',
                'prose_ratio': 0.90,
                'liturgical_focus': 'ritual_explanation',
                'geographic_origin': 'eastern',
                'preservation_quality': 'fair'
            },
            'Satapatha-Br': {
                'genre': 'prose',
                'period': 'late_vedic',
                'chronology': 800,
                'recension': 'madhyandina',
                'veda_branch': 'yajur',
                'text_type': 'brahmana',
                'prose_ratio': 0.98,  # Nearly all prose
                'liturgical_focus': 'ritual_explanation',
                'geographic_origin': 'eastern',
                'preservation_quality': 'excellent'
            },
            'Gopatha-Br': {
                'genre': 'prose',
                'period': 'late_vedic',
                'chronology': 750,
                'recension': 'gopatha',
                'veda_branch': 'atharva',
                'text_type': 'brahmana',
                'prose_ratio': 0.85,
                'liturgical_focus': 'ritual_explanation',
                'geographic_origin': 'eastern',
                'preservation_quality': 'fair'
            },
            
            # UPANISHADS - Mixed, Latest Vedic
            'Brhadaranyaka-Up': {
                'genre': 'prose',
                'period': 'latest_vedic',
                'chronology': 700,
                'recension': 'madhyandina',
                'veda_branch': 'yajur',
                'text_type': 'upanishad',
                'prose_ratio': 0.80,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'excellent'
            },
            'Chandogya-Up': {
                'genre': 'prose',
                'period': 'latest_vedic',
                'chronology': 650,
                'recension': 'chandogya',
                'veda_branch': 'sama',
                'text_type': 'upanishad',
                'prose_ratio': 0.75,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'excellent'
            },
            'Aitareya-Up': {
                'genre': 'prose',
                'period': 'latest_vedic',
                'chronology': 600,
                'recension': 'aitareya',
                'veda_branch': 'rig',
                'text_type': 'upanishad',
                'prose_ratio': 0.70,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'good'
            },
            'Prashna-Up': {
                'genre': 'mixed',
                'period': 'latest_vedic',
                'chronology': 520,
                'recension': 'prashna',
                'veda_branch': 'atharva',
                'text_type': 'upanishad',
                'prose_ratio': 0.65,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'good'
            },
            'Shvetashvatara-Up': {
                'genre': 'mixed',
                'period': 'latest_vedic',
                'chronology': 480,
                'recension': 'shvetashvatara',
                'veda_branch': 'yajur',
                'text_type': 'upanishad',
                'prose_ratio': 0.40,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'excellent'
            },
            'Taittiriya-Up': {
                'genre': 'mixed',
                'period': 'latest_vedic',
                'chronology': 450,
                'recension': 'taittiriya',
                'veda_branch': 'yajur',
                'text_type': 'upanishad',
                'prose_ratio': 0.70,
                'liturgical_focus': 'philosophical',
                'geographic_origin': 'eastern',
                'preservation_quality': 'excellent'
            },
            'Mahabharata': {
                'genre': 'epic',
                'period': 'classical',
                'chronology': 200,
                'recension': 'critical_edition',
                'veda_branch': 'none',
                'text_type': 'epic',
                'prose_ratio': 0.05,
                'liturgical_focus': 'narrative',
                'geographic_origin': 'northern',
                'preservation_quality': 'excellent'
            },
            'Ramayana': {
                'genre': 'epic',
                'period': 'classical',
                'chronology': 300,
                'recension': 'valmiki',
                'veda_branch': 'none',
                'text_type': 'epic',
                'prose_ratio': 0.02,
                'liturgical_focus': 'narrative',
                'geographic_origin': 'northern',
                'preservation_quality': 'excellent'
            },
            'Bhagavata-Purana': {
                'genre': 'purana',
                'period': 'classical',
                'chronology': 800,
                'recension': 'standard',
                'veda_branch': 'none',
                'text_type': 'purana',
                'prose_ratio': 0.30,
                'liturgical_focus': 'devotional',
                'geographic_origin': 'southern',
                'preservation_quality': 'excellent'
            }
        }
    
    def get_metadata_dataframe(self):
        """Convert metadata to DataFrame for analysis"""
        return pd.DataFrame(self.text_metadata).T
    
    def get_categorical_features(self):
        """Get list of categorical predictive features"""
        return ['genre', 'period', 'recension', 'veda_branch', 'text_type', 
                'liturgical_focus', 'geographic_origin', 'preservation_quality']
    
    def get_numerical_features(self):
        """Get list of numerical predictive features"""
        return ['chronology', 'prose_ratio']

class VedicStatisticalModeler:
    """Advanced statistical modeling for Vedic Sanskrit diachronic analysis"""
    
    def __init__(self, analyzer, metadata_handler):
        self.analyzer = analyzer
        self.metadata = metadata_handler
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def prepare_modeling_data(self):
        """Prepare comprehensive dataset for statistical modeling"""
        # Get linguistic features data
        linguistic_df = pd.DataFrame(self.analyzer.results).T
        
        # Get metadata
        metadata_df = self.metadata.get_metadata_dataframe()
        
        # Combine datasets
        combined_df = linguistic_df.join(metadata_df)
        
        # Encode categorical variables
        categorical_cols = self.metadata.get_categorical_features()
        for col in categorical_cols:
            if col in combined_df.columns:
                combined_df[f'{col}_encoded'] = LabelEncoder().fit_transform(combined_df[col])
        
        # Create period ordinal encoding
        period_mapping = {'early_vedic': 1, 'middle_vedic': 2, 'late_vedic': 3, 'latest_vedic': 4}
        combined_df['period_ordinal'] = combined_df['period'].map(period_mapping)
        
        return combined_df
    
    def build_predictive_models(self, target_features=None):
        """Build linear regression models for linguistic features"""
        print("Building Statistical Models for Vedic Sanskrit Features")
        print("=" * 60)
        
        data = self.prepare_modeling_data()
        
        # Default target features (key linguistic variables)
        if target_features is None:
            target_features = [
                'subjunctive_full', 'retroflex_l', 'dual_nominative', 
                'long_compounds', 'philosophical_terms', 'correlatives_ya_ta'
            ]
        
        # Predictive features
        predictor_cols = [
            'chronology', 'prose_ratio', 'period_ordinal',
            'genre_encoded', 'veda_branch_encoded', 'text_type_encoded',
            'liturgical_focus_encoded', 'geographic_origin_encoded'
        ]
        
        # Filter available predictors
        available_predictors = [col for col in predictor_cols if col in data.columns]
        
        results = {}
        
        for target in target_features:
            if target not in data.columns:
                continue
                
            print(f"\nModeling: {target}")
            print("-" * 40)
            
            # Prepare data
            X = data[available_predictors].fillna(0)
            y = data[target].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Try multiple models
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }
            
            target_results = {}
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)-1), 
                                              scoring='r2')
                    
                    # Fit full model
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    
                    # Calculate metrics
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)
                    
                    target_results[name] = {
                        'model': model,
                        'scaler': scaler,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'feature_names': available_predictors
                    }
                    
                    print(f"{name:20s}: R² = {r2:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
                    
                except Exception as e:
                    print(f"{name:20s}: Failed - {str(e)}")
            
            results[target] = target_results
        
        self.models = results
        return results
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        print("\n\nFeature Importance Analysis")
        print("=" * 60)
        
        importance_summary = {}
        
        for target, models in self.models.items():
            print(f"\nTarget: {target}")
            print("-" * 40)
            
            target_importance = {}
            
            for model_name, model_data in models.items():
                model = model_data['model']
                feature_names = model_data['feature_names']
                
                # Extract feature importance based on model type
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances = np.abs(model.coef_)
                else:
                    continue
                
                # Create importance dictionary
                feature_importance = dict(zip(feature_names, importances))
                target_importance[model_name] = feature_importance
                
                # Print top features
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                
                print(f"\n{model_name} - Top Features:")
                for feature, importance in sorted_features:
                    print(f"  {feature:25s}: {importance:.4f}")
            
            importance_summary[target] = target_importance
        
        self.feature_importance = importance_summary
        return importance_summary
    
    def perform_regression_analysis(self, target_feature, detailed=True):
        """Perform detailed regression analysis for a specific feature"""
        print(f"\nDetailed Regression Analysis: {target_feature}")
        print("=" * 60)
        
        data = self.prepare_modeling_data()
        
        if target_feature not in data.columns:
            print(f"Feature {target_feature} not found in data")
            return None
        
        # Prepare variables
        predictor_cols = [
            'chronology', 'prose_ratio', 'period_ordinal',
            'genre_encoded', 'veda_branch_encoded', 'text_type_encoded'
        ]
        
        available_predictors = [col for col in predictor_cols if col in data.columns]
        X = data[available_predictors].fillna(0)
        y = data[target_feature].fillna(0)
        
        # Fit OLS regression
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Statistical tests
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = len(available_predictors)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # F-statistic for overall model significance
        mse_model = np.sum((y_pred - y.mean())**2) / p
        mse_residual = np.sum(residuals**2) / (n - p - 1)
        f_stat = mse_model / mse_residual
        
        print(f"Model Performance:")
        print(f"  R-squared: {r2:.4f}")
        print(f"  Adjusted R-squared: {adjusted_r2:.4f}")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  N observations: {n}")
        
        # Coefficient analysis
        print(f"\nCoefficient Analysis:")
        for i, (feature, coef) in enumerate(zip(available_predictors, model.coef_)):
            # Calculate correlation with target
            corr, p_val = pearsonr(X.iloc[:, i], y)
            print(f"  {feature:25s}: β = {coef:8.4f}, r = {corr:6.3f}, p = {p_val:.3f}")
        
        # Residual analysis
        print(f"\nResidual Analysis:")
        print(f"  Mean residual: {residuals.mean():.6f}")
        print(f"  Residual std: {residuals.std():.4f}")
        print(f"  Durbin-Watson: {self._durbin_watson(residuals):.4f}")
        
        return {
            'model': model,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'f_statistic': f_stat,
            'coefficients': dict(zip(available_predictors, model.coef_)),
            'residuals': residuals,
            'predictions': y_pred,
            'actual': y
        }
    
    def _durbin_watson(self, residuals):
        """Calculate Durbin-Watson statistic for autocorrelation"""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
    
    def cluster_texts_by_features(self, feature_subset=None):
        """Perform hierarchical clustering of texts based on linguistic features"""
        print("\nHierarchical Clustering Analysis")
        print("=" * 50)
        
        data = self.prepare_modeling_data()
        
        # Select features for clustering
        if feature_subset is None:
            linguistic_features = [col for col in data.columns 
                                 if col in self.analyzer.features]
        else:
            linguistic_features = feature_subset
        
        # Prepare clustering data
        cluster_data = data[linguistic_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(cluster_data_scaled, method='ward')
        
        # Get cluster assignments
        cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
        
        # Add cluster labels to data
        data['cluster'] = cluster_labels
        
        # Analyze clusters
        print("Cluster Analysis:")
        for i in range(1, max(cluster_labels) + 1):
            cluster_texts = data[data['cluster'] == i].index.tolist()
            print(f"\nCluster {i}: {cluster_texts}")
            
            # Cluster characteristics
            cluster_data_subset = data[data['cluster'] == i]
            print(f"  Period distribution: {cluster_data_subset['period'].value_counts().to_dict()}")
            print(f"  Text type distribution: {cluster_data_subset['text_type'].value_counts().to_dict()}")
        
        return {
            'linkage_matrix': linkage_matrix,
            'cluster_labels': cluster_labels,
            'clustered_data': data
        }
    
    def principal_component_analysis(self):
        """Perform PCA on linguistic features"""
        print("\nPrincipal Component Analysis")
        print("=" * 50)
        
        data = self.prepare_modeling_data()
        
        # Select linguistic features
        linguistic_features = [col for col in data.columns 
                             if col in self.analyzer.features]
        
        feature_data = data[linguistic_features].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(feature_data_scaled)
        
        # Print explained variance
        print("Explained Variance by Component:")
        cumulative_var = 0
        for i, var in enumerate(pca.explained_variance_ratio_[:10]):
            cumulative_var += var
            print(f"  PC{i+1}: {var:.4f} (cumulative: {cumulative_var:.4f})")
        
        # Feature loadings for first few components
        print("\nFeature Loadings (PC1-PC3):")
        for i in range(min(3, len(pca.components_))):
            print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]:.3f} of variance):")
            
            # Get top positive and negative loadings
            loadings = pca.components_[i]
            feature_loadings = list(zip(linguistic_features, loadings))
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, loading in feature_loadings[:8]:
                print(f"  {feature:25s}: {loading:8.4f}")
        
        return {
            'pca': pca,
            'pca_result': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'feature_names': linguistic_features
        }
    
    def validate_models(self, target_features=None):
        """Comprehensive model validation with diagnostic plots"""
        print("\nModel Validation and Diagnostics")
        print("=" * 60)
        
        if not hasattr(self, 'models') or not self.models:
            print("No models found. Run build_predictive_models() first.")
            return None
        
        validation_results = {}
        
        for target, models in self.models.items():
            print(f"\nValidating models for: {target}")
            print("-" * 40)
            
            target_validation = {}
            
            for model_name, model_data in models.items():
                model = model_data['model']
                r2 = model_data['r2']
                cv_r2 = model_data['cv_r2_mean']
                
                # Model stability (difference between training and CV R²)
                stability = abs(r2 - cv_r2)
                
                # Overfitting indicator
                overfitting = "High" if stability > 0.2 else "Medium" if stability > 0.1 else "Low"
                
                # Model interpretation
                if r2 > 0.7:
                    interpretation = "Strong predictive power"
                elif r2 > 0.4:
                    interpretation = "Moderate predictive power"
                elif r2 > 0.2:
                    interpretation = "Weak predictive power"
                else:
                    interpretation = "Poor predictive power"
                
                target_validation[model_name] = {
                    'stability': stability,
                    'overfitting_risk': overfitting,
                    'interpretation': interpretation,
                    'recommended': stability < 0.15 and r2 > 0.3
                }
                
                print(f"  {model_name:20s}: R²={r2:.3f}, Stability={stability:.3f}, Risk={overfitting}")
            
            validation_results[target] = target_validation
        
        return validation_results
    
    def explain_model_predictions(self, target_feature, text_name):
        """Explain predictions for a specific text"""
        if target_feature not in self.models:
            print(f"No model found for {target_feature}")
            return None
        
        data = self.prepare_modeling_data()
        
        if text_name not in data.index:
            print(f"Text {text_name} not found in data")
            return None
        
        # Get best model (highest CV R²)
        best_model_name = max(self.models[target_feature].keys(), 
                             key=lambda x: self.models[target_feature][x]['cv_r2_mean'])
        
        model_data = self.models[target_feature][best_model_name]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Get text data
        text_data = data.loc[text_name, feature_names].values.reshape(1, -1)
        text_data_scaled = scaler.transform(text_data)
        
        # Make prediction
        prediction = model.predict(text_data_scaled)[0]
        actual = data.loc[text_name, target_feature]
        
        print(f"\nPrediction Explanation: {text_name} - {target_feature}")
        print("=" * 60)
        print(f"Model: {best_model_name}")
        print(f"Predicted: {prediction:.3f}")
        print(f"Actual: {actual:.3f}")
        print(f"Error: {abs(prediction - actual):.3f}")
        
        # Feature contributions (for linear models)
        if hasattr(model, 'coef_'):
            print(f"\nFeature Contributions:")
            contributions = model.coef_ * text_data_scaled[0]
            
            for feature, contrib, value in zip(feature_names, contributions, text_data[0]):
                print(f"  {feature:25s}: {contrib:8.4f} (value: {value:.3f})")
        
        return {
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual),
            'model_used': best_model_name
        }

class VedicStatisticalVisualizer:
    """Visualization functions for statistical modeling results"""
    
    def __init__(self, modeler):
        self.modeler = modeler
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_model_performance(self):
        """Plot model performance comparison"""
        if not hasattr(self.modeler, 'models') or not self.modeler.models:
            print("No models to plot. Run build_predictive_models() first.")
            return
        
        # Prepare performance data
        performance_data = []
        
        for target, models in self.modeler.models.items():
            for model_name, model_data in models.items():
                performance_data.append({
                    'target': target,
                    'model': model_name,
                    'r2': model_data['r2'],
                    'cv_r2': model_data['cv_r2_mean'],
                    'cv_r2_std': model_data['cv_r2_std']
                })
        
        df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. R² comparison by model type
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='model', y='r2', ax=ax1)
        ax1.set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Cross-validation vs training R²
        ax2 = axes[0, 1]
        ax2.scatter(df['r2'], df['cv_r2'], alpha=0.7)
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('Training R²')
        ax2.set_ylabel('Cross-Validation R²')
        ax2.set_title('Model Stability (Training vs CV)', fontsize=14, fontweight='bold')
        
        # 3. Performance by target feature
        ax3 = axes[1, 0]
        target_performance = df.groupby('target')['r2'].mean().sort_values(ascending=False)
        target_performance.plot(kind='bar', ax=ax3)
        ax3.set_title('Average R² by Linguistic Feature', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Linguistic Feature')
        ax3.set_ylabel('Average R²')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Model complexity vs performance
        ax4 = axes[1, 1]
        model_complexity = {
            'Linear Regression': 1,
            'Ridge Regression': 2,
            'Lasso Regression': 2,
            'Random Forest': 4,
            'Gradient Boosting': 5
        }
        
        df['complexity'] = df['model'].map(model_complexity)
        complexity_performance = df.groupby('complexity')['r2'].mean()
        
        ax4.plot(complexity_performance.index, complexity_performance.values, 'o-', linewidth=2)
        ax4.set_xlabel('Model Complexity')
        ax4.set_ylabel('Average R²')
        ax4.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../output/vedic_statistical_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance_heatmap(self):
        """Plot feature importance heatmap across models"""
        if not hasattr(self.modeler, 'feature_importance') or not self.modeler.feature_importance:
            print("No feature importance data. Run analyze_feature_importance() first.")
            return
        
        # Prepare importance matrix
        all_features = set()
        all_models = set()
        
        for target, models in self.modeler.feature_importance.items():
            for model_name, features in models.items():
                all_features.update(features.keys())
                all_models.add(f"{target}_{model_name}")
        
        # Create importance matrix
        importance_matrix = pd.DataFrame(index=sorted(all_features), 
                                       columns=sorted(all_models))
        
        for target, models in self.modeler.feature_importance.items():
            for model_name, features in models.items():
                col_name = f"{target}_{model_name}"
                for feature, importance in features.items():
                    importance_matrix.loc[feature, col_name] = importance
        
        # Fill NaN with 0
        importance_matrix = importance_matrix.fillna(0).astype(float)
        
        # Plot heatmap
        plt.figure(figsize=(20, 12))
        sns.heatmap(importance_matrix, 
                   cmap='YlOrRd', 
                   center=0,
                   annot=False,
                   fmt='.3f',
                   cbar_kws={'label': 'Feature Importance'})
        
        plt.title('Feature Importance Across Models and Targets', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Model_Target Combinations', fontsize=12)
        plt.ylabel('Predictive Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('../output/vedic_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, pca_results):
        """Plot PCA results"""
        pca = pca_results['pca']
        pca_result = pca_results['pca_result']
        
        # Get metadata for coloring
        data = self.modeler.prepare_modeling_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Explained variance
        ax1 = axes[0, 0]
        n_components = min(10, len(pca.explained_variance_ratio_))
        ax1.bar(range(1, n_components + 1), 
                pca.explained_variance_ratio_[:n_components])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
        
        # 2. Cumulative explained variance
        ax2 = axes[0, 1]
        cumulative_var = np.cumsum(pca.explained_variance_ratio_[:n_components])
        ax2.plot(range(1, n_components + 1), cumulative_var, 'o-')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax2.legend()
        
        # 3. PC1 vs PC2 by period
        ax3 = axes[1, 0]
        colors = {'early_vedic': 'red', 'middle_vedic': 'orange', 
                 'late_vedic': 'blue', 'latest_vedic': 'green'}
        
        for period, color in colors.items():
            mask = data['period'] == period
            if mask.any():
                ax3.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                          c=color, label=period, alpha=0.7, s=100)
        
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax3.set_title('PCA: Texts by Historical Period', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PC1 vs PC2 by text type
        ax4 = axes[1, 1]
        text_type_colors = {'samhita': 'red', 'brahmana': 'blue', 'upanishad': 'green'}
        
        for text_type, color in text_type_colors.items():
            mask = data['text_type'] == text_type
            if mask.any():
                ax4.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                          c=color, label=text_type, alpha=0.7, s=100)
        
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax4.set_title('PCA: Texts by Type', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/vedic_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression_diagnostics(self, regression_results, target_feature):
        """Plot regression diagnostic plots"""
        predictions = regression_results['predictions']
        actual = regression_results['actual']
        residuals = regression_results['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(actual, predictions, alpha=0.7)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Actual vs Predicted: {target_feature}', fontweight='bold')
        
        # Add R² text
        r2 = regression_results['r2']
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        ax2.scatter(predictions, residuals, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted', fontweight='bold')
        
        # 3. Q-Q plot of residuals
        ax3 = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals', fontweight='bold')
        
        # 4. Histogram of residuals
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=8, density=True, alpha=0.7, color='skyblue')
        
        # Overlay normal distribution
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                'r-', label='Normal')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Residuals', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'../output/vedic_regression_diagnostics_{target_feature}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clustering_dendrogram(self, clustering_results):
        """Plot hierarchical clustering dendrogram"""
        linkage_matrix = clustering_results['linkage_matrix']
        data = clustering_results['clustered_data']
        
        plt.figure(figsize=(15, 8))
        
        # Create dendrogram
        dendrogram(linkage_matrix, 
                  labels=data.index.tolist(),
                  leaf_rotation=45,
                  leaf_font_size=10)
        
        plt.title('Hierarchical Clustering of Vedic Texts\n(Based on Linguistic Features)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Texts')
        plt.ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig('../output/vedic_text_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()