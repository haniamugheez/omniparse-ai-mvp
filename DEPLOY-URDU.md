# OmniParse AI — deploy karne ka tareeqa

Pehli baar kar rahi hain to bhi 15 minute se zyada nahi lagega. Kuch paisa
nahi lagta, koi card nahi chahiye.

---

## Step 1 — Zip ko kholein

`omniparse-ai.zip` ko Desktop par extract karein. Ek folder banega
`omniparse-ai` jiske andar ye cheezein hongi:

```
app.py
requirements.txt
packages.txt
README.md
omniparse/
sample_data/
tests/
```

---

## Step 2 — Apne computer par chala kar dekh lein (optional, 3 minute)

Deploy se pehle ek baar khud dekhna hamesha behtar hota hai.

PowerShell kholein, ek ek line:

```
cd Desktop\omniparse-ai
```
```
pip install -r requirements.txt
```
```
streamlit run app.py
```

Browser khud khul jayega. Login screen par email aur password pehle se bhare
huye hain — bas **Sign in** dabayein.

Band karne ke liye PowerShell mein `Ctrl + C`.

> Agar `streamlit` command nahi milti to `python -m streamlit run app.py`
> likhein.

---

## Step 3 — GitHub par daalein

Aap ne pehle bhi GitHub use kiya hai, wahi tareeqa hai.

1. https://github.com/new par jayein
2. Repository name: `omniparse-ai`
3. **Public** choose karein (Streamlit ka free plan public repo chahta hai)
4. Baaki sab khali chhor dein — README, .gitignore, license kuch add na karein
5. **Create repository** dabayein

Ab PowerShell mein, folder ke andar, ek ek line:

```
cd Desktop\omniparse-ai
```
```
git init
```
```
git add .
```
```
git commit -m "OmniParse AI MVP"
```
```
git branch -M main
```
```
git remote add origin https://github.com/AAPKA-USERNAME/omniparse-ai.git
```
```
git push -u origin main
```

`AAPKA-USERNAME` ki jagah apna GitHub username likhein.

---

## Step 4 — Streamlit Community Cloud par deploy

1. https://share.streamlit.io par jayein
2. **Continue with GitHub** — wahi account jis par abhi code daala
3. **Create app** → **Deploy a public app from GitHub**
4. Teen khaane bharein:
   - Repository: `AAPKA-USERNAME/omniparse-ai`
   - Branch: `main`
   - Main file path: `app.py`
5. **Deploy** dabayein

Pehli baar 3-5 minute lagte hain (Tesseract aur packages install hote hain).
Screen par logs chalte rahenge — ghabrayein nahi, ye normal hai.

Ban jane ke baad aap ko ek link milega, kuch is tarah:

```
https://omniparse-ai.streamlit.app
```

Ye link kisi ko bhi bhej sakti hain. Hamesha chalta rahega.

---

## Step 5 — Demo kaise dikhayein (4 minute)

1. Link kholein → **Sign in** (email/password pehle se bhare huye hain)
2. **Load sample data** tab → **Everything** → **Load sample invoice data**
3. **Preview the documents** khol kar dikha dein ke ye asli documents hain —
   PDF, scan, aur spreadsheet
4. **▶ Run agent pipeline** dabayein
5. **Audit summary** — 4 documents, 9 records, anomalies, kitne rupay hold par
6. **🚨 Audit report** tab — har finding ke neeche uska hisaab likha hai.
   Ye sab se ahem screen hai:
   - `INV-2026-5102` — "2 × 75,600 = 151,200, lekin invoice 158,200 keh rahi hai"
   - `INV-2025-1003` — "ye invoice number pehle se ledger mein hai"
   - `INV-2026-5005` — "100,000 ki approval limit se sirf 1,500 neeche"
7. **📋 Extracted records** — saaf table, har record ki confidence aur risk
8. **🔗 Agent trace** — kaun sa agent kya kar raha tha, kitne milliseconds mein

Khaas baat jo zaroor batayein: **ek invoice bilkul saaf hai** (`INV-2026-5001`)
aur system us par kuch bhi flag nahi karta. Jo tool har cheez par alarm bajata
hai usay log band kar dete hain — ye nahi karta.

---

## Agar kuch toot jaye

**"ModuleNotFoundError" deploy ke waqt** — `requirements.txt` repo mein push
nahi hui. GitHub par ja kar dekh lein ke file wahan hai.

**Scan wali image par "OCR engine is not installed"** — `packages.txt` repo
mein nahi gayi. Wahi check karein. PDF, CSV aur Excel phir bhi kaam karte
rahenge.

**App "sleeping" ho jaye** — free plan par kuch din na chalne se so jati hai.
Link kholne par khud jaag jati hai, 30 second lagte hain. Demo se 5 minute
pehle ek baar khol lein.

**Code mein kuch badalna ho** — file theek karein, phir:

```
git add .
```
```
git commit -m "fix"
```
```
git push
```

Streamlit khud dobara deploy kar dega.

---

## Ek cheez jo saaf bata dein

Login screen sirf **demonstration** ke liye hai — asli security nahi hai, aur
screen par khud likha hai ke ye placeholder hai. Agar ye kisi asli company ke
documents ke liye use hogi to pehle asli login lagana hoga.

Baqi sab kuch asli hai: documents waqai parhe jate hain, hisaab waqai hota
hai, aur har finding ka hisaab screen par likha hota hai taake koi bhi khud
check kar sake.
