import os
import numpy as np
import pandas as pd
import sys
import glob

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.project.config import DATA_PATH, EMBEDDINGS_PATH


def load_apartments_data():
    files = glob.glob(f"{DATA_PATH}/flats_clean_*.parquet")
    latest_file = sorted(files)[-1]
    df = pd.read_parquet(latest_file)
    return df


def load_embeddings():
    """Загружаем разные файлы для разных типов эмбеддингов"""
    multimodal_path = f"{EMBEDDINGS_PATH}/multimodal_embeddings.npy"
    multimodal = np.load(multimodal_path)
    
    text_path = f"{EMBEDDINGS_PATH}/text_embeddings.npy"
    text_only = np.load(text_path)
    
    return multimodal, text_only


def prepare_numeric_features(df):
    numeric_cols = ['Площадь', 'Количество комнат', 'Этаж', 'Количество этажей в доме']
    
    X = df[numeric_cols].copy()
    X['Этаж_норм'] = X['Этаж'] / X['Количество этажей в доме'].replace(0, 1)
    
    y = df['Цена'].copy()
    
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, valid_mask


def prepare_numeric_text_features(df, text_only_embeddings, valid_mask):
    """Числовые + текст"""
    X_num, y, _ = prepare_numeric_features(df)
    
    valid_indices = valid_mask[valid_mask].index
    text_aligned = text_only_embeddings[valid_indices]
    
    X = np.hstack([X_num, text_aligned])
    
    return X, y


def prepare_all_features(df, multimodal_embeddings, valid_mask):
    """Числовые + текст + изображения"""
    X_num, y, _ = prepare_numeric_features(df)
    
    valid_indices = valid_mask[valid_mask].index
    multimodal_aligned = multimodal_embeddings[valid_indices]
    
    X = np.hstack([X_num, multimodal_aligned])
    
    return X, y


def train_and_evaluate(X, y, model_name, test_size=0.2, cv_folds=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        l2_leaf_reg=2,
        depth=6,
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    
    print(f"\n{model_name}:")
    print(f"   MAE: {mae:,.0f} ₽ ({mae/1_000_000:.2f} млн ₽)")
    print(f"   R²: {r2:.4f}")
    print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   MAPE: {mape:.1f}%")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model,
        'n_features': X.shape[1]
    }


def create_summary_table(results):
    summary = []
    for r in results:
        summary.append({
            'Модель': r['model_name'],
            'Признаков': r['n_features'],
            'MAE (тыс ₽)': f"{r['mae']/1_000_000:.2f}",
            'RMSE (тыс ₽)': f"{r['rmse']/1_000_000:.2f}",
            'R²': f"{r['r2']:.4f}",
            'MAPE (%)': f"{r['mape']:.1f}",
            'CV R²': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"
        })
    
    return pd.DataFrame(summary)


def main():
    df = load_apartments_data()
    
    multimodal_embeddings, text_only_embeddings = load_embeddings()
    
    X_num, y_num, valid_mask = prepare_numeric_features(df)
    print(f"\nЧисловые признаки: {X_num.shape[1]} признаков, {len(X_num)} объектов")
    
    X_text, y_text = prepare_numeric_text_features(df, text_only_embeddings, valid_mask)
    print(f"+ текст: {X_text.shape[1]} признаков")
    
    X_all, y_all = prepare_all_features(df, multimodal_embeddings, valid_mask)
    print(f"+ текст + изображения: {X_all.shape[1]} признаков")
    
    results = []
    
    results.append(train_and_evaluate(X_num, y_num, "1 Только числовые"))
    results.append(train_and_evaluate(X_text, y_text, "2 Числовые + текст"))
    results.append(train_and_evaluate(X_all, y_all, "3 Числовые + текст + изображения"))
    
    df_summary = create_summary_table(results)
    print(df_summary.to_string(index=False))
    
    baseline = results[0]
    for r in results[1:]:
        mae_imp = (baseline['mae'] - r['mae']) / baseline['mae'] * 100
        r2_imp = (r['r2'] - baseline['r2']) / abs(baseline['r2']) * 100 if baseline['r2'] != 0 else 0
        print(f"\n{r['model_name']}:")
        print(f"   MAE: {mae_imp:+.1f}%")
        print(f"   R²: {r2_imp:+.1f}%")


if __name__ == "__main__":
    main()