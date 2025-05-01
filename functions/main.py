# main.py  (functions/)  – v2  
# • görsel “generatedImages/<uid>/…” klasörüne kaydedilir  
# • kullanıcının Firestore’daki coin’i kontrol edilir ve başarıyla
#   tamamlandığında 1 coin düşülür (transaction’lı)  

import os, tempfile, datetime, base64, requests
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage, firestore
from firebase_functions import https_fn, params
import openai


# ──────────────────────────── Firebase / OpenAI ────────────────────────────
SERVICE_KEY = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
cred        = credentials.Certificate(SERVICE_KEY)
try:
    firebase_admin.initialize_app(cred)             # storage + firestore
except ValueError:
    pass                                            # zaten init edilmiş

db          = firestore.client()
OPENAI_KEY  = params.SecretParam("OPENAI_API_KEY")
IMAGE_SIZE  = (1024, 1024)                          # tek noktadan değiştir


def get_bucket():
    return storage.bucket()


# ───────────────────────────── edit_image Cloud Function ───────────────────
@https_fn.on_call(timeout_sec=120, secrets=[OPENAI_KEY], memory=512)
def edit_image(req: https_fn.CallableRequest) -> dict:
    # 0) Kimlik ve parametre kontrolleri
    uid       = req.auth.uid if req.auth else None
    file_path = req.data.get("filePath")
    prompt    = req.data.get("prompt")

    if uid is None:
        raise https_fn.HttpsError("unauthenticated", "Giriş yapmanız gerekiyor.")
    if not file_path or not prompt:
        raise https_fn.HttpsError("invalid-argument", "filePath ve prompt zorunludur.")

    # 1) Coin kontrolü (transaction ile)
    user_ref = db.collection("users").document(uid)

    @firestore.transactional
    def _check_and_decrement(transaction):
        snap = user_ref.get(transaction=transaction)
        coins = snap.get("coin") or 0
        if coins < 1:
            raise https_fn.HttpsError("failed-precondition", "Yetersiz kredi.")
        transaction.update(user_ref, {"coin": coins - 1})

    trx = db.transaction()
    _check_and_decrement(trx)             # başarısızsa exception fırlatır

    # 2) Orijinal dosyayı indir ve 1024×1024 PNG ’e normalize et
    bucket  = get_bucket()
    blob_in = bucket.blob(file_path)
    if not blob_in.exists():
        raise https_fn.HttpsError("not-found", f"{file_path} bulunamadı")

    with tempfile.TemporaryDirectory() as tmp:
        local_in  = os.path.join(tmp, "input.png")
        local_out = os.path.join(tmp, "output.png")

        blob_in.download_to_filename(local_in)

        img = Image.open(local_in).convert("RGBA").resize(IMAGE_SIZE)
        img.save(local_in)

        # 3) OpenAI çağrısı
        client = openai.OpenAI(api_key=OPENAI_KEY.value)
        with open(local_in, "rb") as f_img:
            resp = client.images.edit(
                model="gpt-image-1",
                image=f_img,
                prompt=prompt,
            )

        # 4) Gelen görseli diske yaz (b64 veya URL)
        data_item = resp.data[0]
        if getattr(data_item, "b64_json", None):
            with open(local_out, "wb") as fp:
                fp.write(base64.b64decode(data_item.b64_json))
        elif getattr(data_item, "url", None):
            r = requests.get(data_item.url)
            r.raise_for_status()
            with open(local_out, "wb") as fp:
                fp.write(r.content)
        else:
            raise https_fn.HttpsError("internal", "OpenAI uygun görsel döndürmedi.")

        # 5) Storage’a yükle  → generatedImages/<uid>/...
        out_dir  = f"generatedImages/{uid}/"
        out_name = os.path.basename(file_path)
        out_path = os.path.join(out_dir, out_name)

        blob_out = bucket.blob(out_path)
        blob_out.upload_from_filename(local_out, content_type="image/png")

        signed_url = blob_out.generate_signed_url(
            expiration=datetime.timedelta(days=3650), method="GET"
        )

    # 6) Başarılı cevap
    return {"generatedUrl": signed_url}
