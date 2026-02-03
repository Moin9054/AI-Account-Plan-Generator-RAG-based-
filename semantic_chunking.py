import os 
import docx
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

Claude_api_key = os.environ.get("CLAUDE_KEY")

if not Claude_api_key:
    print("claude api key not found.")
    exit()

Input_file = os.path.join("data", "input", "Knowledge Base.docx")
Output_file = os.path.join("data", "output", "Semantic_chunk.csv")

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None
    
kb_text = read_docx(Input_file)

if kb_text:
    print(f"Successfully read {Input_file}.")

    separator = "||---CHUNK_BREAK---||"

    claude_prompt = f"""
You are an expert at processing and structuring documents.
Your task is read the text and split into logical, contained chunks.

RULES:
1.  A chunks should contain sinle topic.
2.  It can be a single paragraph or multiple paragraphs.
3.  Avoid splitting topic in middle and exch chunks should have full context.
4.  Output *only* the chunks, and separate each chunk with the separator: {separator}
Tthe text:

<document_text>
{kb_text}
</document_text>
"""
    try:
        client = anthropic.Anthropic(api_key=Claude_api_key)

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            system="You are a document processing assistant.",
            messages=[
                {"role": "user", "content": claude_prompt}
            ]
        )

        claude_response = message.content[0].text
        print("received response from Claude.")
    
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        exit()

    try:
        split_chunks = claude_response.split(separator)
        cleaned_chunks = [chunk.strip() for chunk in split_chunks if chunk.strip()]
        
        print(f"Split the text into {len(cleaned_chunks)} chunks.")

        df = pd.DataFrame(cleaned_chunks, columns=["chunk_text"])
        os.makedirs(os.path.dirname(Output_file), exist_ok=True)
        df.to_csv(Output_file, index=False, encoding="utf-8")
        
        print(f"Successfully saved {len(cleaned_chunks)} chunks to {Output_file}.")
        
    except Exception as e:
        print(f"Error processing Claude response or saving to CSV: {e}")
        
else:
    print(f"Could not read {Input_file}.")