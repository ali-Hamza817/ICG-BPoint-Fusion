"""Step 21: build the login-protected website (site_secure/) from the viewer data and figures.

HOW THE LOGIN WORKS (GitHub Pages is a static host: there is no server to check a password):
  * All data (patient JSON) and images (figures, strips, thumbnails) are encrypted with one random 256-bit key K (AES-256-GCM) and only the ciphertext is published.
  * K itself is stored in vault.json, encrypted with a key derived from  <login id>\\n<password>  by PBKDF2-HMAC-SHA256 (600,000 iterations, random salt).
  * The browser derives that key from what the visitor types, decrypts K (AES-GCM authentication fails for a wrong id/password) and decrypts the files locally.
  * The password itself is never stored anywhere in the site or the repository.
LIMITS (be honest in the README): this protects the website's content from casual visitors; it is not server-side security. Anyone can download the ciphertext and try
  passwords offline (a long random password makes that impractical).  The same data/figures also exist unencrypted in other folders of the public repository.
Usage:  python 21_build_secure_site.py [--id advisor] [--password '...']   (no --password = generate a strong one and save it to private/credentials.txt, which is git-ignored)"""
import argparse, base64, hashlib, hmac, io, json, secrets, shutil
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "site_secure"; SRC = ROOT / "site_src" / "index.html"
DATA = ROOT / "deploy_task1" / "data"; FIG = ROOT / "figures" / "task2"; ITER = 600_000
b64 = lambda b: base64.b64encode(b).decode()
ap = argparse.ArgumentParser(); ap.add_argument("--id", default="advisor"); ap.add_argument("--password", default=None); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--key-file", default=str(ROOT / "private" / "site_key.bin")); a = ap.parse_args()
OUT = Path(a.out); KEYFILE = Path(a.key_file)
login = a.id.strip().lower()
if a.password is None:
    alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; pw = "-".join("".join(secrets.choice(alpha) for _ in range(4)) for _ in range(4))
    (ROOT / "private").mkdir(exist_ok=True); (ROOT / "private" / "credentials.txt").write_text(f"login id : {login}\npassword : {pw}\n")
else: pw = a.password
# the data key is kept in a private, git-ignored file and REUSED across rebuilds, so a browser tab opened before an update keeps working;
# delete the key file (or pass another --key-file) to rotate it: tabs opened earlier are then sent back to the login page.
if KEYFILE.exists(): K = KEYFILE.read_bytes()
else: KEYFILE.parent.mkdir(exist_ok=True, parents=True); K = AESGCM.generate_key(bit_length=256); KEYFILE.write_bytes(K)
gcm = AESGCM(K); kid = hashlib.sha256(K).hexdigest()[:16]; bid = secrets.token_hex(6)      # kid: key identity (public), bid: build id (cache busting)

(OUT / "enc").mkdir(parents=True, exist_ok=True)
def put(path, data):                                                  # path = logical name used by the page, e.g. data/patients.json
    # deterministic nonce = HMAC(K, path | sha256(data)): identical plaintext -> identical ciphertext, so unchanged files do not change in git between deploys
    iv = hmac.new(K, b"iv\0" + path.encode() + hashlib.sha256(data).digest(), hashlib.sha256).digest()[:12]
    (OUT / "enc" / (path.replace("/", "__") + ".bin")).write_bytes(iv + gcm.encrypt(iv, data, path.encode()))
def thumb(p, w):
    im = Image.open(p).convert("RGB"); h = round(im.height * w / im.width); buf = io.BytesIO(); im.resize((w, h), Image.LANCZOS).save(buf, "PNG", optimize=True); return buf.getvalue()

n = 0
for f in sorted(DATA.glob("*.json")): put(f"data/{f.name}", f.read_bytes()); n += 1
for f in sorted(FIG.glob("*.png")): put(f"task2/{f.name}", f.read_bytes()); put(f"task2/thumbs/{f.name}", thumb(f, 900)); n += 2
for f in sorted((FIG / "strips").glob("*.png")): put(f"task2/strips/{f.name}", f.read_bytes()); put(f"task2/thumbs/strip_{f.name}", thumb(f, 640)); n += 2
LAB = ROOT / "deploy_task1" / "lab"                                    # data for the in-browser Lab (step 26)
for f in sorted(LAB.glob("*")) if LAB.exists() else []: put(f"lab/{f.name}", f.read_bytes()); n += 1

salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
kek = PBKDF2HMAC(hashes.SHA256(), 32, salt, ITER).derive(f"{login}\n{pw}".encode())
(OUT / "vault.json").write_text(json.dumps(dict(v=2, iter=ITER, salt=b64(salt), iv=b64(iv), ct=b64(AESGCM(kek).encrypt(iv, K, None)), kid=kid, bid=bid)))
shutil.copy(SRC, OUT / "index.html"); (OUT / ".nojekyll").write_text("")
for extra in sorted(list(SRC.parent.glob("*.js")) + list(SRC.parent.glob("*.py"))): shutil.copy(extra, OUT / extra.name)      # lab.js, lab_*.py (public code, loaded on demand)
print(f"encrypted {n} files -> {OUT}; key id {kid}; login id '{login}'" + ("; password saved in private/credentials.txt" if a.password is None else ""))
