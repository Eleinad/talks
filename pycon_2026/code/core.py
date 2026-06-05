import os
from pathlib import Path
import io
import json
from glob import glob
from typing import List, Tuple, Optional, Union, Iterable, Dict
import pickle

from config import BLIP_MODEL_NAME, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT

import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

import logging

import umap
from sklearn.cluster import HDBSCAN


import optuna
import hyperparams_optimization as hyperopt
from functools import partial

from bertTopic_utilities import ensure_ollama_server, stop_ollama_server, map_labels_to_descriptiveTopics_imgs
from prompt_bertopic_label_generation import labels_generation_prompt

logger = logging.getLogger(__name__)

from bertopic.representation import VisualRepresentation 
from bertopic.representation import  OpenAI as BERTopicOpenAI
from bertopic import BERTopic
from openai import OpenAI as OpenAIClient

import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration


# Allowed image file extensions
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

def _list_images(images_path: str) -> List[str]:
    """
    Recursively lists all image files in a directory with allowed extensions.
    Removes duplicates and sorts the list.
    """
    images = []
    for ext in ALLOWED_EXTS:
        images.extend(glob(os.path.join(images_path, f"**/*{ext}"), recursive=True))
    images = sorted(list(dict.fromkeys(images)))  # de-dup, keep order
    return images


def _ensure_ip_index(dimension: int) -> faiss.IndexIDMap:
    base = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap(base)

class ClipFaissIndex:
    def __init__(self, model_name: str = "clip-ViT-B-32", cache_folder: str = "./clip_model"):
        """
        Initializes the CLIP model and sets up variables for the FAISS index.
        """
        # SentenceTransformers CLIP can encode both text and PIL images
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.index: Optional[faiss.Index] = None
        self.image_paths: List[str] = []
        self.dimension: Optional[int] = None
        self.id_to_path: Dict[int, str] = {} # to map id to image path
        self.bertopic_best_hyperparams_filepath: Optional[str] = None
        self.topics_labels_filepath: Optional[str] = None # folder path of bertopic labels mapping



    def generate_clip_embeddings(self, images_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Generates CLIP embeddings for all images in the specified directory.
        Returns the embeddings and the corresponding image paths.
        """
        image_paths = _list_images(images_path)
        if not image_paths:
            raise ValueError(f"No images found under: {images_path}")

        # Batch encode images for efficiency
        images = [Image.open(p).convert("RGB") for p in image_paths]
        emb = self.model.encode(
            images,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # normalizes for cosine similarity
        )
        emb = emb.astype(np.float32)
        return emb, image_paths

    def create_faiss_index(self, embeddings: np.ndarray, image_paths: List[str], output_path: str):
        """
        Creates a FAISS index using the provided embeddings and image paths.
        Saves the index and image paths to disk.
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D (num_images x dim)")
        self.dimension = embeddings.shape[1]

        # Inner product with normalized vectors ≡ cosine similarity
        index = faiss.IndexFlatIP(self.dimension) # IP = Inner Product
        index = faiss.IndexIDMap(index)

        # Assign unique IDs to embeddings
        ids = np.arange(len(embeddings)).astype(np.int64)
        index.add_with_ids(embeddings, ids)

        # Save the index to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        faiss.write_index(index, output_path)

        # Persist mapping as "id\tpath"
        with open(output_path + ".paths", "w", encoding="utf-8") as f:
            for i,p in enumerate(image_paths):
                f.write(f"{i}\t{p}\n")

        self.index = index
        self.image_paths = image_paths[:]
        self.id_to_path = {i: p for i, p in enumerate(image_paths)}


    def load_faiss_index(self, index_path: str):
        """
        Loads a FAISS index and its associated image paths from disk.
        """
        index = faiss.read_index(index_path)
        id_to_path = {}
        image_paths = []

        # Read lines; support legacy format (no IDs) and new "id\tpath"
        with open(index_path + ".paths", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    # New format
                    id_str, path = line.split("\t", 1)
                    try:
                        id_to_path[int(id_str)] = path
                    except ValueError:
                        pass
                else:
                    # Legacy format
                    image_paths.append(line)
        
        if id_to_path:
            self.id_to_path = id_to_path
            # sort by id to build list
            image_paths = [id_to_path[i] for i in sorted(id_to_path.keys())]
        else:
            self.id_to_path = {i: p for i, p in enumerate(image_paths)}

        self.index = index
        self.image_paths = image_paths
        self.dimension = index.d if index.ntotal > 0 else self.dimension

        return index, image_paths

    def _encode_query(
        self, query: Union[str, Image.Image, bytes, io.BytesIO]
    ) -> np.ndarray:
        """
        Encodes a query (text, PIL image, or raw image bytes) into a normalized vector.
        """
        if isinstance(query, (bytes, io.BytesIO)):
            # Handle raw image bytes or stream
            img = Image.open(io.BytesIO(query) if isinstance(query, bytes) else query).convert("RGB")
            vec = self.model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        elif isinstance(query, Image.Image):
            # Handle PIL image
            vec = self.model.encode(query.convert("RGB"), convert_to_numpy=True, normalize_embeddings=True)
        elif isinstance(query, str):
            # Handle text query
            vec = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        else:
            raise TypeError("Unsupported query type. Provide text, PIL.Image, or image bytes/stream.")
        vec = vec.astype(np.float32).reshape(1, -1)
        return vec

    def search(
        self,
        query: Union[str, Image.Image, bytes, io.BytesIO],
        top_k: int = 3,
    ) -> Tuple[List[str], List[float]]:
        """
        Searches the FAISS index for the top-k most similar items to the query.
        Returns the paths and similarity scores of the top-k results.
        """
        if self.index is None or not self.image_paths:
            raise RuntimeError("Index not loaded/built yet.")
        q = self._encode_query(query)
        distances, indices = self.index.search(q, k=top_k)
        # distances are cosine similarities (because we normalized)
        paths = [self.id_to_path.get(int(i)) for i in indices[0] if int(i) != -1 and int(i) in self.id_to_path]
        scores = [float(s) for s, i in zip(distances[0], indices[0]) if int(i) in self.id_to_path]
        return paths, scores

    # add new images to mapping and faiss index
    def add_images(self, image_paths: Iterable[str], index_path: str) -> int:
        """
        Add new images to the FAISS index.

        Parameters
        ----------
        image_paths : Iterable[str]
            An iterable of file paths to images to add. Non-file paths are silently skipped.
        index_path : str
            Path to the FAISS index file to load or initialize.

        Returns
        -------
        int
            Count of newly added images (duplicates against already-indexed paths are skipped).
        """
        image_paths = [p for p in image_paths if os.path.isfile(p)]
        if not image_paths:
            return 0

        existing_paths = set(self.id_to_path.values())
        to_add = [p for p in image_paths if p not in existing_paths]
        if not to_add:
            return 0

        images = []
        for p in to_add:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))
        emb = self.model.encode(images, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        dim = emb.shape[1]
        self._ensure_loaded_or_init(index_path=index_path, dim_hint=dim)
        if self.dimension is not None and self.dimension != dim:
            raise ValueError(f"Embedding dim mismatch: index={self.dimension}, new={dim}")

        # Assign IDs continuing from max existing id + 1 (not len()) to keep IDs stable after deletions
        next_id = (max(self.id_to_path.keys()) + 1) if self.id_to_path else 0
        ids = np.arange(next_id, next_id + len(to_add), dtype=np.int64)

        self.index.add_with_ids(emb, ids)
        for i, p in zip(ids.tolist(), to_add):
            self.id_to_path[int(i)] = p

        # Rebuild legacy list for compatibility (sorted by id)
        self.image_paths = [self.id_to_path[i] for i in sorted(self.id_to_path.keys())]

        # Persist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(index_path + ".paths", "w", encoding="utf-8") as f:
            for i in sorted(self.id_to_path.keys()):
                f.write(f"{i}\t{self.id_to_path[i]}\n")

        return len(to_add)
    
    def _ensure_loaded_or_init(self, index_path: str, dim_hint: int = None):
        """
        If the index is None, try loading from disk. If still absent, initialize a fresh IP index.
        """
        if self.index is None:
            if os.path.exists(index_path):
                self.load_faiss_index(index_path)
            else:
                if dim_hint is None:
                    raise RuntimeError("Cannot initialize a new index without a dimension hint.")
                self.index = _ensure_ip_index(dim_hint)
                self.dimension = dim_hint
                self.image_paths = []


    def add_from_dir(self, images_dir: str, index_path: str) -> int:
        """
        Recursively scans a directory and appends only missing images to the index.
        Returns the number added.
        """
        candidates = _list_images(images_dir)
        return self.add_images(candidates, index_path)

    def remove_images_by_paths(self, paths: List[str], index_path: str) -> int:
        """Remove given file paths from FAISS + in-memory mapping. Returns count removed from index."""
        if self.index is None:
            raise RuntimeError("Index not loaded.")

        # Find IDs for these paths
        path_set = set(paths)
        ids_to_remove = [i for i, p in self.id_to_path.items() if p in path_set]
        if not ids_to_remove:
            return 0

        # Remove from FAISS
        arr = np.array(ids_to_remove, dtype=np.int64)
        self.index.remove_ids(arr)

        # Update mapping
        for i in ids_to_remove:
            self.id_to_path.pop(i, None)
        self.image_paths = [self.id_to_path[i] for i in sorted(self.id_to_path.keys())]

        # Persist updated index + mapping
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(index_path + ".paths", "w", encoding="utf-8") as f:
            for i in sorted(self.id_to_path.keys()):
                f.write(f"{i}\t{self.id_to_path[i]}\n")

        return len(ids_to_remove)
    

    def get_all_embeddings(self):
        '''Get all image embeddings from faiss vector store'''

        if self.index is None:
            raise RuntimeError("Index not loaded.")

        # Get number of vectors
        n = self.index.ntotal

        # The IndexIDMap internally stores a numpy array of IDs
        ids = faiss.vector_to_array(self.index.id_map)
        logger.debug("IDs shape: %s", ids.shape)

        # Extract embeddings
        logger.debug("Reconstructing embeddings...")
        embeddings = np.vstack([self.index.index.reconstruct(i) for i in range(n)])
        logger.debug("Embeddings shape: %s", embeddings.shape)

        return embeddings, n


    def _bertopic_folder(self, index_path: str) -> Path:
        """Return the BERTopic output folder corresponding to the given FAISS index path."""
        return Path(index_path.replace("faiss", "bertTopic")).parent

    def bertTopic_exists(self, index_path: str) -> bool:
        """Check if bertTopic topic_map.json already exists for this FAISS index."""
        bertopic_folder = self._bertopic_folder(index_path)
        topic_map = bertopic_folder / "bertTopic_model" / "topic_map.json"
        if topic_map.exists():
            self.topics_labels_filepath = str(topic_map)
            return True
        return False
    
    def bertTopic_hyperparams_exists(self, index_path: str) -> bool:
        """Check if best UMAP+HDBSCAN params JSON already exists for this FAISS index."""
        bertopic_folder = self._bertopic_folder(index_path)
        params_file = bertopic_folder / "best_umap_hdbscan_params.json"
        if params_file.exists():
            self.bertopic_best_hyperparams_filepath = str(params_file)
            return True
        return False

    def find_best_bertopic_parameters(self, index_path: str):
        """Find best UMAP + HDBSCAN parameters for BERTopic model."""
        if self.index is None:
            self.load_faiss_index(index_path + "vector.index")

        all_embeddings = self.get_all_embeddings()

        # Maximize silhouette score over OPTUNA_N_TRIALS trials with a OPTUNA_TIMEOUT hard timeout
        study = optuna.create_study(study_name="umap_hdbscan_optimization", direction="maximize")
        objective = partial(hyperopt.objective_fn_umap_hdbscan, data=all_embeddings[0])
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)
        umap_hdbscan_best_params = study.best_trial.params

        # Instantiate configured objects so they can be serialized and reused by create_and_apply_best_bertopic_model
        best_umap = umap.UMAP(
            n_neighbors=umap_hdbscan_best_params["n_neighbors"],
            n_components=umap_hdbscan_best_params["n_components"],
            min_dist=umap_hdbscan_best_params["min_dist"],
            metric=umap_hdbscan_best_params["umap_metric"],
        )

        best_hdbscan = HDBSCAN(
            min_cluster_size=umap_hdbscan_best_params["min_cluster_size"],
            min_samples=umap_hdbscan_best_params["min_samples"],
            cluster_selection_epsilon=umap_hdbscan_best_params["cluster_selection_epsilon"],
            metric=umap_hdbscan_best_params["metric"],
        )

        bertopic_folder = self._bertopic_folder(index_path)
        bertopic_folder.mkdir(parents=True, exist_ok=True)

        # Persist raw params as JSON for inspection, and fitted objects as pickles for BERTopic
        with open(bertopic_folder / "best_umap_hdbscan_params.json", "w") as fout:
            json.dump(umap_hdbscan_best_params, fout)

        with open(bertopic_folder / "best_umap.pkl", "wb") as fout:
            pickle.dump(best_umap, fout)

        with open(bertopic_folder / "best_hdbscan.pkl", "wb") as fout:
            pickle.dump(best_hdbscan, fout)

        self.bertopic_best_hyperparams_filepath = str(bertopic_folder / "best_umap_hdbscan_params.json")


    def create_and_apply_best_bertopic_model(self, index_path: str, img_path: str, all_embeddings):
        """Create and apply BERTopic model with best UMAP + HDBSCAN parameters."""
        ensure_ollama_server()

        client = OpenAIClient(base_url="http://localhost:11434/v1", api_key="ollama")
        model = "qwen3:0.6b" #"llama3.2:3b"

        llama_repr = BERTopicOpenAI(
            client,
            model=model,
            prompt=labels_generation_prompt,
            generator_kwargs={"stop": "***"},
        )

        if self.index is None:
            self.load_faiss_index(index_path)

        device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_hf = 0 if torch.cuda.is_available() else -1

        bertopic_folder = self._bertopic_folder(index_path)

        with open(bertopic_folder / "best_umap.pkl", "rb") as fin:
            best_umap = pickle.load(fin)

        with open(bertopic_folder / "best_hdbscan.pkl", "rb") as fin:
            best_hdbscan = pickle.load(fin)

        processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            use_safetensors=True,
            dtype=torch.float16 if device_torch.type == "cuda" else torch.float32,
        ).to(device_torch)

        caption_pipe = pipeline(
            "image-to-text",
            model=blip_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device_hf,
        )

        representation_model = {
            "Visual_Aspect": VisualRepresentation(
                image_to_text_model=caption_pipe,
                batch_size=32,
                image_squares=True,
            ),
            "ollama_labels": llama_repr,
        }

        topic_model = BERTopic(
            umap_model=best_umap,
            hdbscan_model=best_hdbscan,
            calculate_probabilities=True,
            verbose=True,
            representation_model=representation_model,
        )

        all_images = _list_images(img_path)
        topics, probs = topic_model.fit_transform(
            documents=None, embeddings=all_embeddings, images=all_images
        )
        llm_labels = {                                                                                                                                                                                           
            topic_id: labels[0][0].split("\n")[0]                                                                                                                                                                   
            for topic_id, labels in topic_model.get_topics(full=True)["ollama_labels"].items()                                                                                                                       
        }
        topic_model.set_topic_labels(list(llm_labels.values()))

        model_dir = bertopic_folder / "bertTopic_model"
        topic_model.save(
            str(model_dir),
            serialization="safetensors",
            save_embedding_model=False,
            save_ctfidf=True,
        )

        labels_topic_imgs_map = map_labels_to_descriptiveTopics_imgs(topics, all_images, llm_labels)

        with open(model_dir / "topic_map.json", "w") as f:
            json.dump(labels_topic_imgs_map, f, indent=2)

        self.topics_labels_filepath = str(model_dir / "topic_map.json")
        stop_ollama_server()

        return labels_topic_imgs_map, str(model_dir)

    def load_bertopic_clusters_labels_size(self, index_path: str):
        """Load existing BERTopic mapping of clusters and labels."""
        bertopic_folder = self._bertopic_folder(index_path)
        topic_map_path = bertopic_folder / "bertTopic_model" / "topic_map.json"
        with open(topic_map_path, "r") as f:
            labels_topic_imgs_map = json.load(f)
        return labels_topic_imgs_map, str(bertopic_folder / "bertTopic_model")
    
    def update_topic_map_file(self, bertTopic_folder_path: str, new_mapping: Dict):
        try:
            # Convert each TopicEntry to dict before saving to json (new_mapping is Dict[int, TopicEntry])
            serializable_new_mapping = {
                key: entry.model_dump() 
                for key, entry in new_mapping.items()
            }
            with open(f"{os.path.join(bertTopic_folder_path,"topic_map.json")}", "w") as f:
                json.dump(serializable_new_mapping, f, indent=2)
            return True, "Topics mapping file updated correctly."
        except Exception as e:
            return False, str(e)