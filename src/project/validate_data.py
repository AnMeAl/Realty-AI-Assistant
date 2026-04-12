import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
import glob
import os
from datetime import datetime

def create_schema():
    schema = DataFrameSchema(
        columns={
            'Цена': Column(float, checks=[Check.gt(10000), Check.lt(100000000)], nullable=True, coerce=True),
            'Площадь': Column(float, checks=[Check.gt(10), Check.lt(1000)], nullable=True, coerce=True),
            'Количество комнат': Column(int, checks=[Check.in_range(1, 10)], nullable=True, coerce=True),
            'Этаж': Column(int, checks=[Check.gt(0), Check.lt(100)], nullable=True, coerce=True),
            'Количество этажей в доме': Column(int, checks=[Check.gt(0), Check.lt(100)], nullable=True, coerce=True),
            'Адрес': Column(str, nullable=True),
            'Описание': Column(str, nullable=True),
            'S3_изображения': Column(object, nullable=True),
            'Количество_фото': Column(int, checks=[Check.in_range(0, 50)], nullable=True, coerce=True)
        },
        strict=False,
        coerce=True
    )
    return schema

def convert_types(df):
    df = df.copy()
    numeric_cols = ['Цена', 'Площадь', 'Этаж', 'Количество этажей в доме', 'Количество комнат', 'Количество_фото']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_and_filter(df, schema):
    
    valid_mask = pd.Series([True] * len(df))
    
    if 'Площадь' in df.columns:
        valid_mask &= (df['Площадь'].isna()) | ((df['Площадь'] >= 10) & (df['Площадь'] <= 1000))
    
    if 'Количество комнат' in df.columns:
        valid_mask &= (df['Количество комнат'].isna()) | ((df['Количество комнат'] >= 1) & (df['Количество комнат'] <= 10))
    
    if 'Цена' in df.columns:
        valid_mask &= (df['Цена'].isna()) | ((df['Цена'] >= 10000) & (df['Цена'] <= 100000000))
    
    if 'Этаж' in df.columns and 'Количество этажей в доме' in df.columns:
        floor_num = pd.to_numeric(df['Этаж'], errors='coerce')
        total_floors_num = pd.to_numeric(df['Количество этажей в доме'], errors='coerce')
        valid_mask &= (floor_num.isna()) | (total_floors_num.isna()) | (floor_num <= total_floors_num)
        valid_mask &= (floor_num.isna()) | (floor_num <= 100)
        valid_mask &= (total_floors_num.isna()) | (total_floors_num <= 100)
    
    if 'Адрес' in df.columns:
        addr_str = df['Адрес'].astype(str)
        valid_mask &= (addr_str == 'nan') | (addr_str.str.len() >= 3)
    
    valid_df = df[valid_mask].copy()
    
    numeric_cols = ['Цена', 'Площадь', 'Количество комнат', 'Этаж', 'Количество этажей в доме']
    valid_df = valid_df.dropna(subset=numeric_cols, how='all')
    
    try:
        valid_df = schema.validate(df, lazy=True)
        return valid_df
        
    except pa.errors.SchemaError as e:
        print("Найдены ошибки валидации")
        
        valid_mask = pd.Series([True] * len(df))
        
        for col_name, col_schema in schema.columns.items():
            if col_name in df.columns:
                for check in col_schema.checks:
                    try:
                        if hasattr(check, '_element_wise') and check._element_wise:
                            result = df[col_name].apply(lambda x: check._check_fn(x) if pd.notna(x) else True)
                            valid_mask &= result
                        else:
                            result = check(df[col_name])
                            if isinstance(result, pd.Series):
                                valid_mask &= result
                    except Exception:
                        pass

        if 'Этаж' in df.columns and 'Количество этажей в доме' in df.columns:
            floor_num = pd.to_numeric(df['Этаж'], errors='coerce')
            total_num = pd.to_numeric(df['Количество этажей в доме'], errors='coerce')
            valid_mask &= (floor_num.isna()) | (total_num.isna()) | (floor_num <= total_num)
        
        valid_df = df[valid_mask].copy()
        print(f"Отфильтровано {len(df) - len(valid_df)} записей")
        print(f"Осталось {len(valid_df)} валидных записей")
        
        return valid_df

def save_clean_dataset(df, output_dir="data/final"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/flats_clean_{timestamp}.parquet"
    df.to_parquet(filename, index=False)
    return filename

files = glob.glob("data/processed/flats_with_photos_*.csv")
if not files:
    print("Файл не найден!")
    exit(1)

latest_file = sorted(files)[-1]
df = pd.read_csv(latest_file)

schema = create_schema()
valid_df = validate_and_filter(df, schema)

if len(valid_df) > 0:
    save_clean_dataset(valid_df)
else:
    print("Нет валидных данных!")