# app.py
import os
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

from send2trash import send2trash


from core import ClipFaissIndex

from config import CLIP_MODEL_NAME as MODEL_NAME, CLIP_CACHE_DIR as CACHE_DIR, STREAMLIT_ORIGIN

# Initialize FastAPI application
app = FastAPI(title="CLIP Image Search API", version="1.0.0")


# CORS middleware configuration for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[STREAMLIT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the CLIP + FAISS engine
engine = ClipFaissIndex(model_name=MODEL_NAME, cache_folder=CACHE_DIR)

# Request and response models for building and loading the index
class BuildIndexRequest(BaseModel):
    images_path: Optional[str] = None  # Path to images directory
    index_path: Optional[str] = None  # Path to save the index

class BuildIndexResponse(BaseModel):
    images_indexed: int  # Number of images indexed
    index_path: str  # Path to the saved index

# Response models for search results
class SearchResponseItem(BaseModel):
    image_path: str  # Path to the image
    score: float  # Similarity score

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]  # List of search results

class AddIndexRequest(BaseModel):
    images_path: Optional[str] = None      # if provided, scan this folder; else defaults to IMAGES_DIR
    index_path: Optional[str] = None       # optional override of index path

class AddIndexResponse(BaseModel):
    added: int
    total_indexed: int
    index_path: str

# --- Models for image listing ---
class ImageItem(BaseModel):
    path: str   # server filesystem path (for reference)
    url: str    # http url to display (served via /static)
    modified: str    # last modified time (ISO 8601 string)

class ImageListResponse(BaseModel):
    count: int
    page: int
    page_size: int
    items: List[ImageItem]


# --- Models for deleting images ---
class DeleteRequest(BaseModel):
    paths: List[str]                 # absolute or content-relative paths
    index_path: Optional[str] = None # optional override
    images_dir: Optional[str] = None # optional images dir for relative paths

class DeleteResponse(BaseModel):
    removed_from_index: int
    removed_files: int
    errors: Dict[str, str]

class ClusterImagesRequest(BaseModel):
    index_path: Optional[str] = None
    images_path: Optional[str] = None
    embeddings: Optional[List[List[float]]] = None

class TopicEntry(BaseModel):
    llm_label: str
    img_paths: List[str]
    count: int

class ClusterImagesResponse(BaseModel):
    clustering_output: Dict[int, TopicEntry]
    bertTopic_path: str

class UpdateClusterImagesRequest(BaseModel):
    updated_mapping: Dict[int, TopicEntry]
    bertTopic_path: str

class UpdateClusterImagesResponse(BaseModel):
    success: bool
    message: str



# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}  # Return a simple status message

# Endpoint to check the status of the index
@app.get("/index/status")
def index_status():
    return {
        "loaded": engine.index is not None,  # Whether the index is loaded
        "num_items": int(engine.index.ntotal) if engine.index else 0,  # Number of items in the index
        "dimension": int(engine.dimension) if engine.dimension else None,  # Dimension of the index
        "images_path_count": len(engine.image_paths),  # Number of image paths
    }

# Endpoint to build the index from images
@app.post("/index/build", response_model=BuildIndexResponse)
def build_index(payload: BuildIndexRequest):
    images_path = payload.images_path #or IMAGES_DIR  # Use provided or default images path
    index_path = payload.index_path #or INDEX_PATH  # Use provided or default index path
    os.makedirs(os.path.dirname(index_path), exist_ok=True)  # Ensure index directory exists

    # Generate embeddings and create the FAISS index
    embeddings, image_paths = engine.generate_clip_embeddings(images_path)
    engine.create_faiss_index(embeddings, image_paths, index_path)

    return BuildIndexResponse(images_indexed=len(image_paths), index_path=index_path)

# Endpoint to load an existing index
@app.post("/index/load", response_model=BuildIndexResponse)
def load_index(payload: BuildIndexRequest):
    index_path = payload.index_path #or INDEX_PATH  # Use provided or default index path
    if not os.path.exists(index_path):  # Check if the index file exists
        raise HTTPException(status_code=404, detail=f"Index not found at {index_path}")
    engine.load_faiss_index(index_path)  # Load the index
    return BuildIndexResponse(images_indexed=len(engine.image_paths), index_path=index_path)


# Endpoint to perform a search using text or image
@app.post("/search", response_model=SearchResponse)
async def search(
    query: Optional[str] = Form(None),  # Text query
    top_k: int = Form(3),  # Number of top results to return
    image: Optional[UploadFile] = File(None),  # Image file for search
    index_path: Optional[str] = Form(None),  # Index path
):
    if engine.index is None:  # Check if the index is loaded
        # Try to auto-load an existing index if present
        if os.path.exists(index_path):
            engine.load_faiss_index(index_path)
        else:
            raise HTTPException(status_code=400, detail="Index not loaded. Build or load the index first.")

    if (query is None or query.strip() == "") and image is None:  # Ensure at least one input is provided
        raise HTTPException(status_code=400, detail="Provide either a text 'query' or an 'image'.")

    if image is not None:  # Perform image-based search
        content = await image.read()
        paths, scores = engine.search(content, top_k=top_k)
    else:  # Perform text-based search
        paths, scores = engine.search(query.strip(), top_k=top_k)

    # Format the search results
    results = [SearchResponseItem(image_path=p, score=round(float(s), 4)) for p, s in zip(paths, scores)]
    return SearchResponse(results=results)

# New endpoint: append images by scanning a directory (no file upload)
@app.post("/index/add", response_model=AddIndexResponse)
def index_add_scan(payload: AddIndexRequest):
    images_path = payload.images_path #or IMAGES_DIR
    index_path = payload.index_path #or INDEX_PATH

    # Try load existing index if present; otherwise will init on first add
    if engine.index is None and os.path.exists(index_path):
        engine.load_faiss_index(index_path)

    added = engine.add_from_dir(images_path, index_path)
    total = len(engine.image_paths)
    return AddIndexResponse(added=added, total_indexed=total, index_path=index_path)

@app.get("/images/list", response_model=ImageListResponse)
def list_images(
    page: int = 1,
    page_size: int = 48,
    images_dir: Optional[str] = None
):
    root = Path(images_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    all_paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and "bertTopic_model" not in p.parts])
    total = len(all_paths)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    slice_paths = all_paths[start:end]

    items = []
    for p in slice_paths:
        # Instead of /static, build URL to the dynamic image server endpoint
        url = f"/image?folder={p.parent.as_posix()}&image_name={p.name}"
        stat = p.stat()
        btime = getattr(stat, "st_birthtime", stat.st_ctime)
        btime_human = datetime.fromtimestamp(btime).isoformat(sep=" ", timespec="seconds")
        items.append(ImageItem(path=str(p), url=url, modified=btime_human))

    return ImageListResponse(
        count=total,
        page=page,
        page_size=page_size,
        items=items
    )


@app.post("/images/delete", response_model=DeleteResponse)
def delete_images(payload: DeleteRequest):
    idx_path = payload.index_path # or INDEX_PATH
    paths_to_delete = payload.paths # or IMAGES_DIR

    # Make sure index is in memory if file exists
    if engine.index is None:
        if os.path.exists(idx_path):
            engine.load_faiss_index(idx_path)
        else:
            raise HTTPException(status_code=400, detail="Index not loaded and not found on disk.")

    # Remove files from disk
    removed_files = 0
    errors: Dict[str, str] = {}
    for p in paths_to_delete:
        p = Path(p) # normalizing path string to Path
        try:
            if os.path.exists(p):
                send2trash(p)
                removed_files += 1
        except Exception as e:
            errors[p] = str(e)
        try:
            # Remove from FAISS
            removed_idx = engine.remove_images_by_paths(paths_to_delete, idx_path)
        except Exception as e:
            errors[p] = str(e)

    return DeleteResponse(
        removed_from_index=removed_idx,
        removed_files=removed_files,
        errors=errors
    )

@app.post("/images/cluster", response_model=ClusterImagesResponse)
def cluster_images(payload: ClusterImagesRequest):
    index_path = payload.index_path
    images_path = payload.images_path

    # # Try load existing index if present; otherwise will init on first add
    if engine.index is None and os.path.exists(index_path):
        engine.load_faiss_index(index_path)
    
    embeddings, n_embeddings = engine.get_all_embeddings()

    check_bertopic_exists = engine.bertTopic_exists(index_path)
    check_best_hyperparams_exists = engine.bertTopic_hyperparams_exists(index_path)

    clusters_labels_size, bertopic_folder_path = {}, ""

    if check_bertopic_exists:
        clusters_labels_size, bertopic_folder_path = engine.load_bertopic_clusters_labels_size(index_path)
    else:
        if not check_best_hyperparams_exists:
            engine.find_best_bertopic_parameters(index_path) # stop after hyperparam does not change for n iterations
        clusters_labels_size, bertopic_folder_path = engine.create_and_apply_best_bertopic_model(index_path, images_path, embeddings)

    if not clusters_labels_size:
        raise HTTPException(status_code=500, detail="BertTopic clustering failed.")

    return ClusterImagesResponse(clustering_output=clusters_labels_size, bertTopic_path=bertopic_folder_path)

@app.post("/images/cluster/update")
def update_clustering_topics_images_map(payload: UpdateClusterImagesRequest):
    bertTopic_folder_path = payload.bertTopic_path
    new_mapping = payload.updated_mapping

    updated, msg = engine.update_topic_map_file(bertTopic_folder_path, new_mapping)

    if not updated:
        return HTTPException(status_code=500, detail=msg)
    else:
        return UpdateClusterImagesResponse(
            success = updated,
            message = msg
        )