# streamlit_app.py
import os
import requests
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import urllib.parse
import math


from config import API_URL, DEFAULT_IMAGES_DIR

CLUSTERING_HELPER_TEXT = {"done": "Clustering has already been performed. Check results on 'Images Clusters' tab!",
                          "not_done": "Cluster indexed images for better search performance"}

if "selected_images_txt_search" not in st.session_state:
    st.session_state.selected_images_txt_search = set()   # store paths of selected images to be deleted

if "selected_images_img_search" not in st.session_state:
    st.session_state.selected_images_img_search = set()   # store paths of selected images to be deleted

if "selected_images_clustering" not in st.session_state:
    st.session_state.selected_images_clustering = set()

if "images_dir" not in st.session_state:
    st.session_state.images_dir = DEFAULT_IMAGES_DIR

if "index_path" not in st.session_state:
    st.session_state.index_path = f"./faiss/{st.session_state.images_dir[2:]}/vector.index"

if "results_txt_search" not in st.session_state:
    st.session_state.results_txt_search = []

if "results_img_search" not in st.session_state:
    st.session_state.results_img_search = []

if "results_clustering_images" not in st.session_state:
    st.session_state.results_clustering_images = {}

if "cluster_mapping_path" not in st.session_state:
    st.session_state.cluster_mapping_path = ""

# to change selection state when an image is selected/deselected
def _toggle_selection(path: str, key: str, sess_type: str):
    """Sync a checkbox's value into the persistent set."""
    if st.session_state.get(key, False):
        st.session_state[sess_type].add(path)
    else:
        st.session_state[sess_type].discard(path)

# Configure the Streamlit app's appearance and layout
st.set_page_config(page_title="CLIP Image Search", page_icon="🔎", layout="wide")
st.title("🔎 CLIP Image Search (Text & Image)")

# Sidebar for index controls and image uploads
with st.sidebar:
    st.header("⚙️ Index Controls")
    # Input fields for specifying the images directory and index path on the server
    images_dir = st.text_input("Images directory on server", DEFAULT_IMAGES_DIR).replace('\\','/')
    index_path = f"./faiss/{images_dir[2:]}/vector.index"
    st.session_state.images_dir = images_dir
    st.session_state.index_path = index_path
    st.text(f"Faiss index path: {index_path}")

    # Buttons for building and loading the index
    colA, colB, colC = st.columns(3)
    if colA.button("Build index", help=f"Create FAISS index from content of '{images_dir}' folder"):
        try:
            with st.spinner("Building FAISS index..."):
                r = requests.post(f"{API_URL}/index/build", json={"images_path": images_dir, "index_path": index_path})
            if r.ok:
                st.success(f"Indexed {r.json()['images_indexed']} images.")
            else:
                st.error(r.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend — is it running? ({e})")

    if colB.button("Load index", help=f"Load FAISS index from '{index_path}' path"):
        try:
            r = requests.post(f"{API_URL}/index/load", json={"index_path": index_path})
            if r.ok:
                st.success(f"Loaded index with {r.json()['images_indexed']} images.")
            else:
                st.error(r.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend — is it running? ({e})")

    # Extract clustering condition into named variables for readability
    _clustering_done = len(st.session_state.get("results_clustering_images", {})) != 0
    _clustering_result = st.session_state.get("results_clustering_images", {})
    _bertopic_path = _clustering_result.get("bertTopic_path", "")
    if _bertopic_path:
        _derived = os.path.join(
            *_bertopic_path.replace("\\", "/").split("/")[:-1]
        ).replace("bertTopic", images_dir.split("/")[0]).replace("\\", "/")
    else:
        _derived = ""
    _same_dir = _bertopic_path != "" and _derived in images_dir
    _cluster_btn_disabled = _clustering_done and _same_dir

    if colC.button("Cluster images", help=CLUSTERING_HELPER_TEXT["done"] if _clustering_done else CLUSTERING_HELPER_TEXT["not_done"],
                   disabled=_cluster_btn_disabled):
        try:
            with st.spinner("Running BERTopic clustering pipeline — this may take several minutes..."):
                r = requests.post(
                    f"{API_URL}/images/cluster",
                    json={"index_path": index_path, "images_path": images_dir},
                    timeout=3600,
                )
            if r.ok:
                data = r.json()
                st.session_state.results_clustering_images = data
                st.success(f"Clustering completed. {len(data['clustering_output'].keys())} clusters formed.")
                st.success(f"BertTopic model path: {data['bertTopic_path']}")
                st.session_state.cluster_mapping_path = data.get("bertTopic_path", "")
                st.rerun()  # Refresh to show clustering results
            else:
                st.error(r.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend — is it running? ({e})")

    # Buttons for adding and uploading new images to the index
    col1, col2 = st.columns(2)
    if col1.button("Scan folder for new images and add them to index"):
        try:
            r = requests.post(f"{API_URL}/index/add", json={"images_path": images_dir, "index_path": index_path})
            if r.ok:
                js = r.json()
                st.success(f"Added {js['added']} new images to FAISS index. Total indexed: {js['total_indexed']}.")
            else:
                st.error(r.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend — is it running? ({e})")

# Main section for search functionality
st.subheader("Search")
tab1, tab2, tab3, tab4 = st.tabs(["🔤 Text Query", "🖼️ Image Query", "🔖 Images Clusters", "📂 Browse images"])

# Tab for text-based search
with tab1:
    # Input field for text query and slider for top K results
    text_query = st.text_input("Enter a text prompt (e.g., 'a brown dog running on grass')", "dog")
    top_k = st.slider("Top K", 1, 24, 6, step=1)
    if st.button("Search by Text", type="primary"):
        try:
            with st.spinner("Searching..."):
                data = {"query": text_query, "top_k": str(top_k), "index_path": index_path}
                r = requests.post(f"{API_URL}/search", data=data)
            if r.ok:
                st.session_state.results_txt_search = r.json()["results"]
                if not st.session_state.results_txt_search:
                    st.info("No results.")
            else:
                st.error(r.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend — is it running? ({e})")
    if st.session_state.results_txt_search:
        selected = []  # List of selected images for deletion
        # Display the search results in a grid
        cols = st.columns(min(4, len(st.session_state.results_txt_search)))
        for i, res in enumerate(st.session_state.results_txt_search):
            c = cols[i % len(cols)]
            c.image(res["image_path"], caption=f"score={res['score']:.4f}\n{res['image_path']}")

            # The checkbox key must be stable/unique. Initialize it with current selection state.
            key = f"sel_txt_{res['image_path']}"
            default_checked = res["image_path"] in st.session_state.selected_images_txt_search
            if key not in st.session_state:
                st.session_state[key] = default_checked

            # Checkbox to select for deletion (key must be unique)
            if c.checkbox("Select", key=key, on_change=_toggle_selection, args=(res["image_path"], key, "selected_images_txt_search" )):
                st.session_state.selected_images_txt_search.add(res["image_path"])

        # Action row
        st.markdown("---")
        # ---- Delete action uses the persistent set ----
        to_delete = sorted(st.session_state.selected_images_txt_search)
        left, mid, right = st.columns([1,2,2])
        if left.button("Delete selected", key="delete_from_txt", type="primary", disabled=(len(to_delete) == 0)):
            try:
                r = requests.post(f"{API_URL}/images/delete", json={"paths": to_delete, "index_path": index_path, "images_dir": images_dir}, timeout=60)
                if r.ok:
                    js = r.json()
                    # Remove deleted ones from selection, reflect in checkboxes
                    for p in to_delete:
                        st.session_state.selected_images_txt_search.discard(p)
                        st.session_state.pop(f"sel_txt_{p}", None)
                    st.success(
                        f"Deleted {js['removed_files']} files, "
                        f"removed {js['removed_from_index']} from index."
                    )
                    if js.get("errors"):
                        st.warning(f"Some files failed: {js['errors']}")
                    # Optionally refresh the list to reflect removals
                    st.session_state.results_txt_search = [d for d in st.session_state.results_txt_search if d["image_path"] not in to_delete]
                    st.rerun()                    
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Delete failed: {e}")

# Tab for image-based search
with tab2:
    # File uploader for query image and slider for top K results
    qimg = st.file_uploader("Upload a query image", type=["jpg","jpeg","png","bmp","tiff","gif"], key="img_query")
    if qimg:
        st.image(qimg, caption="Uploaded Image Preview", width=200)

    top_k2 = st.slider("Top K", 1, 24, 6, step=1, key="topk2")
    if st.button("Search by Image", type="primary", key="imgbtn"):
        if qimg is None:
            st.warning("Please upload an image.")
        else:
            try:
                with st.spinner("Searching..."):
                    files = {"image": (qimg.name, qimg.read(), qimg.type)}
                    data = {"top_k": str(top_k2), "index_path": index_path}
                    r = requests.post(f"{API_URL}/search", files=files, data=data)
                if r.ok:
                    st.session_state.results_img_search = r.json()["results"]
                    if not st.session_state.results_img_search:
                        st.info("No results.")
                else:
                    st.error(r.text)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend — is it running? ({e})")
    if st.session_state.results_img_search:
        selected = []  # List of selected images for deletion
        # Display the search results in a grid
        cols = st.columns(min(4, len(st.session_state.results_img_search)))
        for i, res in enumerate(st.session_state.results_img_search):
            c = cols[i % len(cols)]
            c.image(res["image_path"], caption=f"score={res['score']:.4f}\n{res['image_path']}")

            # The checkbox key must be stable/unique. Initialize it with current selection state.
            key = f"sel_img_{res['image_path']}"
            default_checked = res["image_path"] in st.session_state.selected_images_img_search
            if key not in st.session_state:
                st.session_state[key] = default_checked

            # Checkbox to select for deletion (key must be unique)
            if c.checkbox("Select", key=key, on_change=_toggle_selection, args=(res["image_path"], key, "selected_images_img_search" )):
                st.session_state.selected_images_img_search.add(res["image_path"])

        # Action row
        st.markdown("---")
        # ---- Delete action uses the persistent set ----
        to_delete = sorted(st.session_state.selected_images_img_search)
        left, mid, right = st.columns([1,2,2])
        if left.button("Delete selected", key="delete_from_img", type="primary", disabled=(len(to_delete) == 0)):
            try:
                r = requests.post(f"{API_URL}/images/delete", json={"paths": to_delete, "index_path": index_path, "images_dir": images_dir}, timeout=60)
                if r.ok:
                    js = r.json()
                    # Remove deleted ones from selection, reflect in checkboxes
                    for p in to_delete:
                        st.session_state.selected_images_img_search.discard(p)
                        st.session_state.pop(f"sel_img_{p}", None)
                    st.success(
                        f"Deleted {js['removed_files']} files, "
                        f"removed {js['removed_from_index']} from index."
                    )
                    if js.get("errors"):
                        st.warning(f"Some files failed: {js['errors']}")
                    # Optionally refresh the list to reflect removals
                    st.session_state.results_img_search = [d for d in st.session_state.results_img_search if d["image_path"] not in to_delete]
                    st.rerun()
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Delete failed: {e}")

# Tab for images clustering
with tab3:
    st.markdown("### 🔖 Image Clustering with BERTOPIC")

    if not _clustering_done or not _same_dir:
        st.info("Clustering has not been performed yet. Please use the 'Cluster images' button in the sidebar to cluster indexed images.")
    else:
        clustering_data = st.session_state.get("results_clustering_images")
        clusters = clustering_data.get("clustering_output", {})
        thumbnails_dir = os.path.join(clustering_data.get("bertTopic_path",{}),"images")

        # --- Summary ---
        bertopic_path = clustering_data.get("bertTopic_path")
        total_clusters = len(clusters.keys())
        total_images = sum(c.get("count", len(c.get("img_paths", []))) for c in clusters.values())

        st.markdown(
            f"- **Total clusters:** `{total_clusters}`  \n"
            f"- **Total images in clusters:** `{total_images}`"
        )
        if bertopic_path:
            st.markdown(f"- **BERTOPIC model path:** `{bertopic_path}`")

        st.markdown("---")

        # --- Controls for visualization ---
        col_order, col_cols = st.columns([2, 1])
        sort_by = col_order.selectbox(
            "Order clusters by",
            options=["Cluster id (ascending)", "Cluster size (descending)"],
            index=1,
        )
        images_per_row = col_cols.slider("Images per row", 2, 8, 4)

        # Sort clusters
        if sort_by == "Cluster size (descending)":
            sorted_clusters = sorted(
                clusters.items(),
                key=lambda kv: kv[1].get("count", len(kv[1].get("img_paths", []))),
                reverse=True,
            )
        else:
            # sort by numeric cluster id if possible, else lexicographically
            def _cluster_key(kv):
                cid = kv[0]
                try:
                    return int(cid)
                except (TypeError, ValueError):
                    return cid
            sorted_clusters = sorted(clusters.items(), key=_cluster_key)

        # --- Show clusters with selectable images ---

        for cluster_id, cluster_info in sorted_clusters:
            label = (cluster_info.get("llm_label") or "").strip() or "(no label)"
            img_paths = cluster_info.get("img_paths", [])
            count = cluster_info.get("count", len(img_paths))

            header = f"Cluster {cluster_id} – {label} ({count} images)"
            with st.expander(header, expanded=False):

                # --- TOP ROW: thumbnail + label/info ---
                top_left, top_right = st.columns([1, 3])

                # Try to get representative image path (e.g. 0.jpg, 1.jpg, ...)
                rep_img_path = None
                if thumbnails_dir is not None:
                    # assume filenames are "<cluster_id>.*"
                    prefix = str(cluster_id)
                    try:
                        files = [
                            f for f in os.listdir(thumbnails_dir)
                            if f.startswith(prefix)
                        ]
                        if files:
                            rep_img_path = os.path.join(thumbnails_dir, files[0])
                    except FileNotFoundError:
                        pass

                # Left: thumbnail
                with top_left:
                    if rep_img_path is not None:
                        st.image(
                            rep_img_path,
                            # use_container_width=True,
                            width="stretch",
                            caption="Representative",
                        )
                    else:
                        st.caption("No thumbnail available")

                # Right: label chip + basic info
                with top_right:
                    st.markdown(
                        f"<span style='padding:4px 8px; border-radius:999px; "
                        f"background-color:rgba(0,0,0,0.05); font-size:0.85rem;'>"
                        f"{label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Cluster ID:** `{cluster_id}`")
                    st.write(f"**Images in cluster:** `{count}`")

                st.write("")  # small spacer


                if not img_paths:
                    st.warning("No images found for this cluster.")
                    continue

                num_images = len(img_paths)
                num_rows = math.ceil(num_images / images_per_row)

                for row in range(num_rows):
                    cols = st.columns(images_per_row)
                    for col_idx in range(images_per_row):
                        idx = row * images_per_row + col_idx
                        if idx >= num_images:
                            break
                        img_path = img_paths[idx]
                        with cols[col_idx]:
                            st.image(
                                img_path,
                                width="stretch",
                                caption=os.path.basename(img_path),
                            )

                            # Checkbox key must be stable/unique
                            key = f"sel_cluster_{img_path}"
                            default_checked = (
                                img_path in st.session_state.selected_images_clustering
                            )
                            if key not in st.session_state:
                                st.session_state[key] = default_checked

                            if st.checkbox(
                                "Select",
                                key=key,
                                on_change=_toggle_selection,
                                args=(img_path, key, "selected_images_clustering"),
                            ):
                                st.session_state.selected_images_clustering.add(img_path)

        # --- Global delete action for clustering images ---
        st.markdown("---")
        to_delete = sorted(st.session_state.selected_images_clustering)
        left, mid, right = st.columns([1, 2, 2])

        if left.button(
            "Delete selected (clusters)",
            key="delete_from_clusters",
            type="primary",
            disabled=(len(to_delete) == 0),
        ):
            try:
                r = requests.post(
                    f"{API_URL}/images/delete",
                    json={
                        "paths": to_delete,
                        "index_path": index_path,
                        "images_dir": images_dir,
                    },
                )
                if r.ok:
                    js = r.json()

                    # Remove deleted ones from selection + checkbox state
                    for p in to_delete:
                        st.session_state.selected_images_clustering.discard(p)
                        st.session_state.pop(f"sel_cluster_{p}", None)

                    # Update clustering results to reflect deletions
                    clustering_data = st.session_state.results_clustering_images
                    clustering_output = clustering_data.get("clustering_output", {})
                    for cid, cinfo in clustering_output.items():
                        old_paths = cinfo.get("img_paths", [])
                        new_paths = [p for p in old_paths if p not in to_delete]
                        cinfo["img_paths"] = new_paths
                        cinfo["count"] = len(new_paths)
                    st.session_state.results_clustering_images["clustering_output"] = (
                        clustering_output
                    )

                    st.success(
                        f"Deleted {js['removed_files']} files, "
                        f"removed {js['removed_from_index']} from index."
                    )
                    if js.get("errors"):
                        st.warning(f"Some files failed: {js['errors']}")
                    try:
                        r_upd = requests.post(                    
                            f"{API_URL}/images/cluster/update",
                            json={
                                "bertTopic_path": st.session_state.results_clustering_images["bertTopic_path"],
                                "updated_mapping": st.session_state.results_clustering_images["clustering_output"],
                            },
                        )

                        if r_upd.ok:
                            st.success(r_upd.json()["message"]) 
                            st.rerun()
                        else:
                            st.error(r_upd.json()["message"])
                    except Exception as e_upd:
                        st.error(f"Update after deletion failed: {e_upd}")
                else:
                    st.error(r.text)

            except Exception as e:
                st.error(f"Delete failed: {e}")



# Tab for image listing
with tab4:
    st.write("Browse the images currently available on the server.")
    colA, colB, colC = st.columns([1,1,2])
    page_size = colA.selectbox("Page size", [12, 24, 48, 96], index=2)
    page = colB.number_input("Page", min_value=1, value=1, step=1)
    refresh = colC.button("Refresh list")

    # Call the API
    params = {"page": page, "page_size": page_size, "images_dir": images_dir}
    try:
        r = requests.get(f"{API_URL}/images/list", params=params, timeout=30)
        if not r.ok:
            st.error(r.text)
        else:
            data = r.json()
            total = data["count"]
            items = data["items"]

            st.caption(f"Total images: {total} • Showing {len(items)} (page {data['page']})")

            if not items:
                st.info("No images found. Upload some in the sidebar or check your images directory.")
            else:
                # Show in a flexible grid
                cols_per_row = 6 if page_size >= 48 else 4
                cols = st.columns(cols_per_row)
                for i, it in enumerate(items):
                    c = cols[i % cols_per_row]
                    # Build absolute URL in case API is on a different host/port
                    # If API_URL ends with /, strip it
                    api_base = API_URL.rstrip("/")
                    url = it["url"]
                    if url.startswith("/"):
                        url = f"{api_base}{url}"

                    # Image
                    c.image(it['path'], width="stretch")

                    # Info (path + modified)
                    c.caption(f"{it['path']}\n🕒 {it['modified']}")

    except Exception as e:
        st.error(f"Failed to fetch image list: {e}")