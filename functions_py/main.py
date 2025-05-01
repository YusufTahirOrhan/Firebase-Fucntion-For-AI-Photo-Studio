# functions/main.py
import os, uuid, base64, tempfile, datetime, requests
from typing import Tuple
from PIL import Image

import firebase_admin
from firebase_admin import credentials, storage, firestore
from firebase_functions import https_fn, params
import openai

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
cred = credentials.Certificate(SERVICE_ACCOUNT)
try:
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()
OPENAI_KEY = params.SecretParam("OPENAI_API_KEY")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE     : Tuple[int,int] = (1024, 1024)
COIN_COST_EDIT               = 1

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_bucket():
    return storage.bucket()

def user_doc(uid: str):
    return db.collection("users").document(uid)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Purchase–based coin top-up
# ──────────────────────────────────────────────────────────────────────────────
@https_fn.on_call(timeout_sec=60, secrets=[OPENAI_KEY])
def add_coins(req: https_fn.CallableRequest) -> dict:
    uid    = req.auth.uid if req.auth else None
    amount = req.data.get("amount")
    if not uid or not isinstance(amount, int) or amount <= 0:
        raise https_fn.HttpsError("invalid-argument", "You must provide a positive integer amount.")
    user_doc(uid).update({"coin": firestore.Increment(amount)})
    return {"status": "ok", "added": amount}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Edit an image – check & deduct coins transactionally
# ──────────────────────────────────────────────────────────────────────────────
@https_fn.on_call(timeout_sec=120, memory=512, secrets=[OPENAI_KEY])
def edit_image(req: https_fn.CallableRequest) -> dict:
    uid       = req.auth.uid if req.auth else None
    file_path = req.data.get("filePath")
    prompt    = req.data.get("prompt")

    if not uid:
        raise https_fn.HttpsError("unauthenticated", "Authentication required.")
    if not file_path or not prompt:
        raise https_fn.HttpsError("invalid-argument", "filePath and prompt are required.")

    blob_in = get_bucket().blob(file_path)
    if not blob_in.exists():
        raise https_fn.HttpsError("not-found", f"{file_path} not found.")

    # ——————————————
    # Transactional coin deduction
    # ——————————————
    doc_ref    = user_doc(uid)
    transaction = db.transaction()

    @firestore.transactional
    def deduct_coin(txn, ref):
        snap    = ref.get(transaction=txn)
        current = snap.get("coin") or 0
        if current < COIN_COST_EDIT:
            raise https_fn.HttpsError("failed-precondition", "Insufficient coins.")
        # decrement by COIN_COST_EDIT
        txn.update(ref, {"coin": current - COIN_COST_EDIT})

    deduct_coin(transaction, doc_ref)

    # ——————————————
    # OpenAI image edit flow
    # ——————————————
    with tempfile.TemporaryDirectory() as tmp:
        local_in  = os.path.join(tmp, "input.png")
        local_out = os.path.join(tmp, "output.png")

        blob_in.download_to_filename(local_in)
        img = Image.open(local_in).convert("RGBA").resize(IMAGE_SIZE)
        img.save(local_in)

        client = openai.OpenAI(api_key=OPENAI_KEY.value)
        with open(local_in, "rb") as f_img:
            resp = client.images.edit(
                model="gpt-image-1",
                image=f_img,
                prompt=prompt,
                size=f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            )

        # base64 veya URL senaryosu
        if getattr(resp.data[0], "b64_json", None):
            img_bytes = base64.b64decode(resp.data[0].b64_json)
        elif getattr(resp.data[0], "url", None):
            r = requests.get(resp.data[0].url, timeout=30)
            r.raise_for_status()
            img_bytes = r.content
        else:
            raise https_fn.HttpsError("internal", "OpenAI did not return image data.")

        with open(local_out, "wb") as fp:
            fp.write(img_bytes)

        out_path = f"generatedImages/{uid}/{uuid.uuid4().hex}.png"
        blob_out = get_bucket().blob(out_path)
        blob_out.upload_from_filename(local_out, content_type="image/png")

    signed_url = blob_out.generate_signed_url(
        expiration=datetime.timedelta(days=3650), method="GET"
    )
    return {"generatedUrl": signed_url}
