import os, json, time, tempfile, asyncio
from pathlib import Path

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

# Chroma imports
import chromadb
from chromadb.config import Settings as ChromaSettings

# PostgresML import
import psycopg2

from .element import Inputs, Settings

IDLE_TIMEOUT = 3.0

def _add_in_batches(
    col, docs, metas, embs, *, id_prefix="t", max_batch=None
):
    if not docs:
        return
    n = len(docs)
    if max_batch is None:
        max_batch = 2000
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        ids = [f"{id_prefix}{k}" for k in range(i, j)]
        col.add(
            documents=docs[i:j],
            metadatas=metas[i:j],
            embeddings=embs[i:j],
            ids=ids,
        )

def build_stores_chroma(payload, client, max_batch):
    t0 = time.time()
    txt_col = client.get_or_create_collection("text")
    _add_in_batches(
        txt_col,
        payload.get("txt_docs", []),
        payload.get("txt_metadatas", []),
        payload.get("txt_embeddings", []),
        max_batch=max_batch,
        id_prefix="t",
    )
    # images if any…
    img_embs = payload.get("img_embeddings", [])
    if img_embs:
        img_col = client.get_or_create_collection("images")
        _add_in_batches(
            img_col,
            payload.get("img_docs", []),
            payload.get("img_metadatas", []),
            img_embs,
            max_batch=max_batch,
            id_prefix="i",
        )
    print(f"[VectorStore][Chroma] Done in {(time.time()-t0):.1f}s")

def build_stores_postgresml(payload, conn, table_name):
    t0 = time.time()
    docs = payload.get("txt_docs", []) or []
    metas = payload.get("txt_metadatas", []) or []
    embs = payload.get("txt_embeddings", []) or []

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
    for i, (doc, meta, emb) in enumerate(zip(docs, metas, embs)):
        rec_id = f"t{i}"
        cur.execute(
            f"""
            INSERT INTO {table_name} (id, document, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                rec_id,
                doc,
                json.dumps(meta, ensure_ascii=False),
                emb,
            ),
        )
    conn.commit()
    cur.close()
    print(f"[VectorStore][PostgresML] Inserted {len(docs)} rows into {table_name} in {(time.time()-t0):.1f}s")

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
                    _, frame = await asyncio.wait_for(
                        ait.__anext__(), timeout=IDLE_TIMEOUT
                    )
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

            # Chromadb
            if store_type == "chromadb":
                db_dir = Path(settings.vector_db_folder_path.value or "").expanduser()
                db_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(db_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                build_stores_chroma(payload, client, settings.max_batch_size.value)

            # PostgresML
            else:
                host   = settings.pgml_host.value
                port   = settings.pgml_port.value
                db     = settings.pgml_db.value
                user   = settings.pgml_user.value
                pwd    = settings.pgml_password.value or None
                table  = settings.postgres_table_name.value

                # connect and write
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=pwd,
                    dbname=db,
                )
                try:
                    build_stores_postgresml(payload, conn, table)
                finally:
                    conn.close()
                conn.close()

            # cleanup
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
            version="0.26.0",
            description="Writes embeddings to Chroma or PostgresML",
        ),
        run_func=writer.run,
    )
)

