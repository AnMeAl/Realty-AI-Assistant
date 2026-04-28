import os
import sys
import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.project.config import (
    DATA_PATH, EMBEDDINGS_PATH, INDEX_PATH
)
from src.project.embeddings import get_multimodal_embedding_for_apartment
from src.project.embeddings import load_embedding_models, unload_models


def load_apartments_data():
    files = glob.glob(f"{DATA_PATH}/flats_clean_*.parquet")
    if not files:
        raise Exception(f"Нет данных в {DATA_PATH}")
    
    latest_file = sorted(files)[-1]
    df = pd.read_parquet(latest_file)
    return df


def precompute_embeddings(df):
    import faiss
    
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)
    
    text_model, clip_model, clip_processor = load_embedding_models()
    
    all_multimodal_embeddings = []
    metadata_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Прогресс"):
        if pd.isna(row['Цена']):
            continue
        
        emb = get_multimodal_embedding_for_apartment(
            row, 
            text_model, 
            clip_model, 
            clip_processor,
            get_images=True
        )
        all_multimodal_embeddings.append(emb)
        
        metadata_list.append({
            'id': row.get('id', f"apartment_{idx}"),
            'price': float(row['Цена']),
            'area': float(row['Площадь']) if pd.notna(row['Площадь']) else 0,
            'rooms': int(row['Количество комнат']) if pd.notna(row['Количество комнат']) else 0,
            'floor': int(row['Этаж']) if pd.notna(row['Этаж']) else 0,
            'total_floors': int(row['Количество этажей в доме']) if pd.notna(row['Количество этажей в доме']) else 0,
            'address': str(row['Адрес'])[:200] if pd.notna(row['Адрес']) else "",
            'description': str(row.get('Описание', ''))[:300],
            'image_urls': row.get('S3_изображения', []) if row.get('S3_изображения') else []
        })
    
    embeddings_matrix = np.vstack(all_multimodal_embeddings).astype('float32')
    np.save(f"{EMBEDDINGS_PATH}/multimodal_embeddings.npy", embeddings_matrix)
    
    dimension = embeddings_matrix.shape[1]
    faiss.normalize_L2(embeddings_matrix)
    
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings_matrix)
    
    faiss.write_index(faiss_index, f"{INDEX_PATH}/multimodal_index.faiss")
    
    with open(f"{INDEX_PATH}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    unload_models()
    
    return embeddings_matrix, metadata_list, faiss_index


def main():
    df = load_apartments_data()
    precompute_embeddings(df)

if __name__ == "__main__":
    main()