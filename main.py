from fastapi import FastAPI, Form
from rag import query_similarity_search
from chunking import process_file_by_type
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.post("/process_file/")
async def process_file(file_path: str = Form(..., description="Path to the file to be processed")):
    try:
        await process_file_by_type(file_path)
        
        return {"message": f"File processed successfully from path: {file_path}"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/query/")
async def query(query: str = Form(..., description="Query to perform similarity search")):
    try:
        results = query_similarity_search(query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
