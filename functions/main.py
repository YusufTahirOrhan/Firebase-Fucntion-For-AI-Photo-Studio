# functions/main.py
"""
Firebase Python Cloud Functions
───────────────────────────────
1.  Auth trigger  →  Creates a user profile in /users with starter coins.
2.  Callable  add_coins()  →  Adds coins after a verified in-app purchase.
3.  Callable  edit_image() →  • Checks coin balance in a Firestore transaction
                             • Calls OpenAI GPT-Image-1 to edit the picture
                             • Stores the result under  generatedImages/{uid}/
                             • Returns a long-lived signed URL
"""

import os
import uuid
import base64
import tempfile
import datetime
import requests
from typing import Tuple

from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage, firestore
from firebase_functions import https_fn, auth_fn, params
import openai


# ────────────────────────────────
# Firebase / OpenAI configuration
# ────────────────────────────────
SERVICE_ACCOUNT = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
cred = credentials.Certificate(SERVICE_ACCOUNT)

try:
    firebase_admin.initialize_app(cred)   # default app + default bucket
except ValueError:
    pass                                  # already initialized on hot-reloads

db          = firestore.client()
OPENAI_KEY  = params.SecretParam("OPENAI_API_KEY")


# ────────────────────────────────
# Constants
# ────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (1024, 1024)   # supported square size
COIN_START                   = 5             # free coins on sign-up
COIN_COST_EDIT               = 1             # price per edit


# ────────────────────────────────
# Helpers
# ────────────────────────────────
def bucket():
    return storage.bucket()

def user_ref(uid: str):
    return db.collection("users").document(uid)


# ────────────────────────────────
# 1. Auth trigger – create profile
# ────────────────────────────────
@auth_fn.on_create()
def create_user_profile(e: auth_fn.AuthCreateEvent) -> None:
    user_ref(e.uid).set({
        "email":        e.email,
        "displayName":  e.display_name,
        "createdAt":    firestore.SERVER_TIMESTAMP,
        "coin":         COIN_START,
    })


# ────────────────────────────────
# 2. Add coins (callable)
# ────────────────────────────────
@https_fn.on_call(timeout_sec=60, secrets=[OPENAI_KEY])
def add_coins(req: https_fn.CallableRequest) -> dict:
    uid = req.auth.uid if req.auth else None
    amount = req.data.get("amount")

    if not uid or not isinstance(amount, int) or amount <= 0:
        raise https_fn.HttpsError("invalid-argument", "Positive integer 'amount' required")

    user_ref(uid).update({"coin": firestore.Increment(amount)})
    return {"status": "ok", "added": amount}


# ────────────────────────────────
# 3. Edit image (callable)
# ────────────────────────────────
@https_fn.on_call(
    timeout_sec=120,
    memory=512,
    secrets=[OPENAI_KEY],
)
def edit_image(req: https_fn.CallableRequest) -> dict:
    uid       = req.auth.uid if req.auth else None
    file_path = req.data.get("filePath")
    prompt    = req.data.get("prompt")

    if not uid:
        raise https_fn.HttpsError("unauthenticated", "User must be signed in")
    if not file_path or not prompt:
        raise https_fn.HttpsError("invalid-argument", "'filePath' and 'prompt' required")

    src_blob = bucket().blob(file_path)
    if not src_blob.exists():
        raise https_fn.HttpsError("not-found", f"{file_path} not found")

    # ----- Firestore transaction: spend coin -----
    def spend_coin(tx: firestore.Transaction):
        snap = user_ref(uid).get(transaction=tx)
        coins = snap.get("coin") or 0
        if coins < COIN_COST_EDIT:
            raise https_fn.HttpsError("failed-precondition", "Insufficient coins")
        tx.update(user_ref(uid), {"coin": coins - COIN_COST_EDIT})

    db.transaction(spend_coin)

    # ----- Download source, call OpenAI, save output -----
    with tempfile.TemporaryDirectory() as tmp:
        local_in  = os.path.join(tmp, "input.png")
        local_out = os.path.join(tmp, "output.png")

        src_blob.download_to_filename(local_in)
        Image.open(local_in).convert("RGBA").resize(IMAGE_SIZE).save(local_in)

        client = openai.OpenAI(api_key=OPENAI_KEY.value)
        with open(local_in, "rb") as f_img:
            oa_resp = client.images.edit(
                model="gpt-image-1",
                image=f_img,
                prompt=prompt,
                size=f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            )

        # Either base64 or URL depending on account settings
        if getattr(oa_resp.data[0], "b64_json", None):
            img_bytes = base64.b64decode(oa_resp.data[0].b64_json)
        elif getattr(oa_resp.data[0], "url", None):
            img_bytes = requests.get(oa_resp.data[0].url, timeout=30).content
        else:
            raise https_fn.HttpsError("internal", "OpenAI returned no image")

        with open(local_out, "wb") as fp:
            fp.write(img_bytes)

        dest_path = f"generatedImages/{uid}/{uuid.uuid4().hex}.png"
        dest_blob = bucket().blob(dest_path)
        dest_blob.upload_from_filename(local_out, content_type="image/png")

    signed_url = dest_blob.generate_signed_url(
        expiration=datetime.timedelta(days=3650),
        method="GET",
    )

    return {"generatedUrl": signed_url}
