import numpy as np
import pandas as pd
from PIL import Image
import torch
import io
import boto3
from botocore.client import Config
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from src.project.config import (
    S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, BUCKET_NAME,
    TEXT_MODEL_NAME, CLIP_MODEL_NAME, MAX_IMAGES_PER_APARTMENT
)

DEVICE = 'mps'

_text_model = None
_clip_model = None
_clip_processor = None
_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
    return _s3_client


def load_embedding_models():
    global _text_model, _clip_model, _clip_processor
    
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        _clip_model = _clip_model.to(DEVICE)
        _clip_model.eval()
    
    return _text_model, _clip_model, _clip_processor


def unload_models():
    global _text_model, _clip_model, _clip_processor
    _text_model = None
    _clip_model = None
    _clip_processor = None
    
    if DEVICE == 'mps':
        torch.mps.empty_cache()


def get_text_embedding(text: str) -> np.ndarray:
    model, _, _ = load_embedding_models()
    embedding = model.encode([text[:500]])[0]
    return embedding.astype('float32')


def get_image_embedding_from_url(s3_uri: str) -> np.ndarray:
    _, clip_model, clip_processor = load_embedding_models()
    s3_client = get_s3_client()
    
    try:
        key = s3_uri.replace(f"s3://{BUCKET_NAME}/", "")
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = clip_processor(images=image, return_tensors="pt")
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_embedding = image_features.cpu().numpy()[0]
        
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        return image_embedding.astype('float32')
        
    except Exception as e:
        print(f"Ошибка загрузки {s3_uri[:80]}: {e}")
        return np.zeros(512, dtype='float32')


def get_multimodal_embedding_for_apartment(row, text_model, clip_model, clip_processor, get_images=True) -> np.ndarray:
    embeddings = []
    
    address = row.get('Адрес', '')
    description = row.get('Описание', '')
    if pd.isna(address):
        address = ''
    if pd.isna(description):
        description = ''
    
    text = f"{address} {description}".strip()
    if not text:
        text = "Квартира без описания"
    
    text_emb = text_model.encode([text[:500]])[0]
    embeddings.append(text_emb.astype('float32'))
    
    if get_images:
        s3_uris = row.get('S3_изображения', [])
        if s3_uris and not pd.isna(s3_uris):
            s3_client = get_s3_client()
            image_embeddings = []
            
            for uri in s3_uris[:MAX_IMAGES_PER_APARTMENT]:
                try:
                    key = uri.replace(f"s3://{BUCKET_NAME}/", "")
                    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
                    image_data = response['Body'].read()
                    image = Image.open(io.BytesIO(image_data))
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    inputs = clip_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        image_features = clip_model.get_image_features(**inputs)
                        img_emb = image_features.cpu().numpy()[0]
                        img_emb = img_emb / np.linalg.norm(img_emb)
                        image_embeddings.append(img_emb.astype('float32'))
                except Exception as e:
                    continue
            
            if image_embeddings:
                avg_img_emb = np.mean(image_embeddings, axis=0)
                embeddings.append(avg_img_emb)
    
    multimodal_emb = np.concatenate(embeddings)
    multimodal_emb = multimodal_emb / np.linalg.norm(multimodal_emb)
    
    return multimodal_emb


def get_query_multimodal_embedding(query_text: str) -> np.ndarray:
    text_emb = get_text_embedding(query_text)
    text_emb = text_emb / np.linalg.norm(text_emb)
    return text_emb.reshape(1, -1).astype('float32')