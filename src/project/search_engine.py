import os
import json
import re
import numpy as np
import faiss
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.project.config import INDEX_PATH, DEFAULT_TOP_K


@dataclass
class SearchResult:
    """Результат поиска"""
    id: str
    price: float
    area: float
    rooms: int
    floor: int
    total_floors: int
    address: str
    similarity: float
    image_urls: List[str]
    description: str = ""


class SearchEngine:
    """Поисковый движок на FAISS"""
    
    def __init__(self):
        self.index = None
        self.metadata = []
        self.is_loaded = False
    
    def load(self):
        """Загрузка FAISS индекса и метаданных"""
        index_path = f"{INDEX_PATH}/multimodal_index.faiss"
        metadata_path = f"{INDEX_PATH}/metadata.json"
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("Индекс не найден")
            return False
        
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.is_loaded = True
        return True
    
    def search(self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K,
               price_range: Optional[Tuple[float, float]] = None,
               area_range: Optional[Tuple[float, float]] = None,
               rooms: Optional[List[int]] = None) -> List[SearchResult]:
        """
        Простой поиск (только семантика)
        """
        if not self.is_loaded:
            return []
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(query_norm, top_k * 3)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            if price_range:
                if meta['price'] < price_range[0] or meta['price'] > price_range[1]:
                    continue
            
            if area_range:
                if meta['area'] < area_range[0] or meta['area'] > area_range[1]:
                    continue
            
            if rooms:
                if meta['rooms'] not in rooms:
                    continue
            
            results.append(SearchResult(
                id=meta['id'],
                price=meta['price'],
                area=meta['area'],
                rooms=meta['rooms'],
                floor=meta['floor'],
                total_floors=meta['total_floors'],
                address=meta['address'],
                similarity=float(score),
                image_urls=meta.get('image_urls', []),
                description=meta.get('description', '')
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    
    def search_hybrid(self, query_embedding: np.ndarray, 
                      query_text: str,
                      top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        """
        Гибридный поиск: семантика + жесткие числовые фильтры
        
        Args:
            query_embedding: эмбеддинг запроса
            query_text: исходный текст запроса (для извлечения числовых параметров)
            top_k: количество результатов
        """
        if not self.is_loaded:
            return []
        
        rooms = self._extract_rooms_from_text(query_text)
        price_range = self._extract_price_range_from_text(query_text)
        area_range = self._extract_area_range_from_text(query_text)
        floor_min = self._extract_floor_from_text(query_text)
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(query_norm, top_k * 10)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            if rooms is not None and meta['rooms'] != rooms:
                continue
            
            if floor_min is not None and meta['floor'] < floor_min:
                continue
            
            if price_range is not None:
                if meta['price'] < price_range[0] or meta['price'] > price_range[1]:
                    continue
            
            if area_range is not None:
                if meta['area'] < area_range[0] or meta['area'] > area_range[1]:
                    continue
            
            results.append(SearchResult(
                id=meta['id'],
                price=meta['price'],
                area=meta['area'],
                rooms=meta['rooms'],
                floor=meta['floor'],
                total_floors=meta['total_floors'],
                address=meta['address'],
                similarity=float(score),
                image_urls=meta.get('image_urls', []),
                description=meta.get('description', '')
            ))
            
            if len(results) >= top_k:
                break
        
        if len(results) == 0 and rooms is not None:
            print("Ничего не найдено с фильтром комнат, пробую без фильтра...")
            return self.search_hybrid(query_embedding, query_text, top_k,
                                      price_range, area_range, rooms=None, floor_min=floor_min)
        
        return results

    
    def _extract_rooms_from_text(self, text: str) -> Optional[int]:
        """Извлекает количество комнат из текста запроса"""
        text = text.lower()
        
        if re.search(r'однокомнатн', text):
            return 1
        if re.search(r'двухкомнатн', text):
            return 2
        if re.search(r'тр[её]хкомнатн', text):
            return 3
        if re.search(r'четырехкомнатн', text):
            return 4
        
        match = re.search(r'(\d+)\s*-?\s*(?:комн|к)', text)
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_price_range_from_text(self, text: str) -> Optional[Tuple[float, float]]:
        """Извлекает ценовой диапазон из текста (в рублях)"""
        text = text.lower()
        
        match = re.search(r'(?:до|не дороже|меньше)\s*(\d+(?:[.,]\d+)?)\s*(?:тыс|тысяч)', text)
        if match:
            price = float(match.group(1).replace(',', '.'))
            if 'тыс' in match.group(0):
                price *= 1_000
            else:
                price *= 1_000_000
            return (0, price)
        
        match = re.search(r'от\s*(\d+(?:[.,]\d+)?)\s*(?:тыс|тысяч)?\s*до\s*(\d+(?:[.,]\d+)?)\s*(?:тыс|тысяч)', text)
        if match:
            min_price = float(match.group(1).replace(',', '.')) * 1_000_000
            max_price = float(match.group(2).replace(',', '.')) * 1_000_000
            return (min_price, max_price)
        
        return None
    
    def _extract_area_range_from_text(self, text: str) -> Optional[Tuple[float, float]]:
        """Извлекает диапазон площади из текста (в м²)"""
        text = text.lower()
        
        match = re.search(r'площад[ьи]\s*(?:от\s*)?(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)', text)
        if match:
            return (float(match.group(1).replace(',', '.')), float(match.group(2).replace(',', '.')))
        
        match = re.search(r'до\s*(\d+(?:[.,]\d+)?)\s*(?:кв\.?м|метров|м2)', text)
        if match:
            return (0, float(match.group(1).replace(',', '.')))
        
        match = re.search(r'от\s*(\d+(?:[.,]\d+)?)\s*(?:кв\.?м|метров|м2)', text)
        if match:
            return (float(match.group(1).replace(',', '.')), float('inf'))
        
        return None
    
    def _extract_floor_from_text(self, text: str) -> Optional[int]:
        """Извлекает предпочтения по этажу"""
        text = text.lower()
        
        if re.search(r'высок\w*\s*этаж', text):
            return 5
        
        if re.search(r'не\s*перв\w*\s*этаж|не\s*1\s*этаж', text):
            return 2
        
        if re.search(r'верхн\w*\s*этаж', text):
            return 7
        
        match = re.search(r'(\d+)-?\s*этаж', text)
        if match:
            return int(match.group(1))
        
        return None


search_engine = SearchEngine()