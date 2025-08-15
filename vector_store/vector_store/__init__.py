# vector_store/__init__.py
import os, json, time, tempfile, asyncio
from pathlib import Path

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

import chromadb
from chromadb.config import Settings as ChromaSettings

import psycopg2

from .element import Inputs, Settings

IDLE_TIMEOUT = 3.0

def _add_in_batches(col, docs, metas, embs, *, id_prefix="t", max_batch=None):
    if not docs:
        return
    n = len(docs)
    if max_batch is None:
        max_batch = 2000
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        ids = [f"{id_prefix}{k}" for k in range(i, j)]
        col.add(documents=docs[i:j], metadatas=metas[i:j], embeddings=embs[i:j], ids=ids)

def _base_name(settings):
    return settings.postgres_table_name.value or "default"

def _dedup_by_key(docs, metas, embs, key_fn):
    seen = set()
    od, om, oe = [], [], []
    for d, m, e in zip(docs, metas, embs):
        try:
            k = key_fn(m)
        except Exception:
            k = None
        if k is None or k in seen:
            continue
        seen.add(k)
        od.append(d); om.append(m); oe.append(e)
    return od, om, oe

def build_stores_chroma(payload, client, max_batch, base):
    t0 = time.time()
    text_name   = f"{base}_text"
    images_name = f"{base}_images"

    # Text
    txt_col = client.get_or_create_collection(text_name)
    _add_in_batches(
        txt_col,
        payload.get("txt_docs", []) or [],
        payload.get("txt_metadatas", []) or [],
        payload.get("txt_embeddings", []) or [],
        max_batch=max_batch,
        id_prefix="t",
    )

    # Images (image-derived text and image stubs)
    img_txt_docs   = payload.get("img_txt_docs", []) or []
    img_txt_metas  = payload.get("img_txt_metadatas", []) or []
    img_txt_embs   = payload.get("img_txt_embeddings", []) or []

    img_stub_docs  = payload.get("img_stub_docs", []) or []
    img_stub_metas = payload.get("img_stub_metadatas", []) or []
    img_stub_embs  = payload.get("img_stub_embeddings", []) or []

    if img_txt_docs or img_stub_docs:
        img_col = client.get_or_create_collection(images_name)

        docs_all  = img_txt_docs  + img_stub_docs
        metas_all = img_txt_metas + img_stub_metas
        embs_all  = img_txt_embs  + img_stub_embs
        docs_all, metas_all, embs_all = _dedup_by_key(
            docs_all, metas_all, embs_all,
            key_fn=lambda m: (m.get("image_path"), m.get("source"), m.get("page"))
        )
        _add_in_batches(img_col, docs_all, metas_all, embs_all, max_batch=max_batch, id_prefix="i")

    # Optional legacy pixels
    raw_img_embs = payload.get("img_embeddings", []) or []
    if raw_img_embs:
        pix_name = f"{base}_image_pixels"
        pix_col = client.get_or_create_collection(pix_name)
        _add_in_batches(
            pix_col,
            payload.get("img_docs", []) or [],
            payload.get("img_metadatas", []) or [],
            raw_img_embs,
            max_batch=max_batch,
            id_prefix="p",
        )

    print(f"[VectorStore][Chroma] Done in {(time.time()-t0):.1f}s → {text_name}, {images_name}")

def _ensure_pg_table(conn, table_name):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
      CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        document TEXT,
        metadata JSONB,
        embedding VECTOR
      );
    """)
    conn.commit()
    cur.close()

def _insert_pg_rows(conn, table_name, docs, metas, embs, id_prefix="t"):
    if not docs:
        return 0
    cur = conn.cursor()
    for i, (doc, meta, emb) in enumerate(zip(docs, metas, embs)):
        rec_id = f"{id_prefix}{i}"
        cur.execute(
            f"""
            INSERT INTO {table_name} (id, document, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (rec_id, doc, json.dumps(meta, ensure_ascii=False), emb),
        )
    conn.commit()
    cur.close()
    return len(docs)

def build_stores_postgresml(payload, conn, base):
    t0 = time.time()
    tbl_text   = f"{base}_text"
    tbl_images = f"{base}_images"

    _ensure_pg_table(conn, tbl_text)
    _ensure_pg_table(conn, tbl_images)

    n1 = _insert_pg_rows(conn, tbl_text,
                         payload.get("txt_docs", []) or [],
                         payload.get("txt_metadatas", []) or [],
                         payload.get("txt_embeddings", []) or [],
                         id_prefix="t")

    # Merge image-derived text and image stubs
    docs_all  = (payload.get("img_txt_docs", []) or []) + (payload.get("img_stub_docs", []) or [])
    metas_all = (payload.get("img_txt_metadatas", []) or []) + (payload.get("img_stub_metadatas", []) or [])
    embs_all  = (payload.get("img_txt_embeddings", []) or []) + (payload.get("img_stub_embeddings", []) or [])

    # Dedupe
    docs_all, metas_all, embs_all = _dedup_by_key(
        docs_all, metas_all, embs_all,
        key_fn=lambda m: (m.get("image_path"), m.get("source"), m.get("page"))
    )

    n2 = _insert_pg_rows(conn, tbl_images, docs_all, metas_all, embs_all, id_prefix="i")

    # Optional legacy pixels
    raw_img_embs = payload.get("img_embeddings", []) or []
    if raw_img_embs:
        tbl_pix = f"{base}_image_pixels"
        _ensure_pg_table(conn, tbl_pix)
        _insert_pg_rows(conn, tbl_pix,
                        payload.get("img_docs", []) or [],
                        payload.get("img_metadatas", []) or [],
                        raw_img_embs,
                        id_prefix="p")

    print(f"[VectorStore][PostgresML] Inserted {n1} text rows; {n2} image rows into {tbl_images} in {(time.time()-t0):.1f}s")

class StoreWriter:
    async def run(self, process: Process):
        settings = process.settings
        processed_any = False
        ait = process.inputs.wait_for_frame()

        while True:
            try:
                if not processed_any:
                    _, frame = await ait.__anext__()
                else:
                    _, frame = await asyncio.wait_for(ait.__anext__(), timeout=IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                break
            except StopAsyncIteration:
                break

            other = getattr(frame, "other_data", {}) or {}
            path = other.get("file_path", "").strip()
            if not path or not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            store_type = settings.vector_store_type.value
            base = _base_name(settings)

            if store_type == "chromadb":
                db_dir = Path(settings.vector_db_folder_path.value or "").expanduser()
                db_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False))
                build_stores_chroma(payload, client, settings.max_batch_size.value, base)
            else:
                host   = settings.pgml_host.value
                port   = settings.pgml_port.value
                db     = settings.pgml_db.value
                user   = settings.pgml_user.value
                pwd    = settings.pgml_password.value or None

                conn = psycopg2.connect(host=host, port=port, user=user, password=pwd, dbname=db)
                try:
                    build_stores_postgresml(payload, conn, base)
                finally:
                    conn.close()

            # Cleanup temp files
            for p in payload.get("prev_tmp_files", []) + [path]:
                try:
                    os.remove(p)
                except:
                    pass

            processed_any = True

        print("VectorStore: finished; exiting.")

writer = StoreWriter()

process = CreateElement(
    Process(
        inputs=Inputs(),
        settings=Settings(),
        metadata=ProcessMetadata(
            id="2a7a0b6a-7b84-4c57-8f1c-store000003",
            name="vector_store",
            displayName="MM - Vector Store",
            version="0.30.0",
            description="Writes to Chroma/PostgresML using <base>_text and <base>_images (image text + stubs)."
        ),
        run_func=writer.run,
    )
)


