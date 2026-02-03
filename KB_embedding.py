import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import voyageai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("VOYAGE_API_KEY")

if not api_key:
    raise ValueError("VOYAGE_API_KEY not found in .env file")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment")


try:
    csv_path = os.path.join('Semantic_chunk.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    vo = voyageai.Client(api_key=api_key)
    
    print("Generating Embeddings...")
    embeddings = vo.embed(
        df['chunk_text'].tolist(), 
        model="voyage-3", 
        input_type="document"
    ).embeddings
    
    df['embedding'] = embeddings

    print("Connecting to Database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id bigserial PRIMARY KEY,
            content text,
            embedding vector(1024)
        );
    """)

    print("Inserting data...")
    data_to_insert = [
        (row['chunk_text'], row['embedding']) 
        for _, row in df.iterrows()
    ]

    insert_query = "INSERT INTO knowledge_chunks (content, embedding) VALUES %s"
    execute_values(cur, insert_query, data_to_insert)

    conn.commit()
    print("SUCCESS: Database is populated and ready.")

except Exception as e:
    print(f"ERROR: {e}")
finally:
    if 'conn' in locals(): conn.close()