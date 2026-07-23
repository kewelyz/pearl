# ⛏️ Pearl (PRL) Miner — H100 / H200

Skrip lengkap untuk mining **Pearl (PRL)**, sebuah blockchain Layer-1 dengan
konsensus **Proof-of-Useful-Work**: alih-alih hashing SHA-256, GPU Anda
menjalankan perkalian matriks besar (inference LLM lewat plugin vLLM). Hasil
komputasinya "berguna" untuk AI sekaligus mengamankan jaringan.

> Sumber perintah: repo resmi [`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl).

---

## ⚠️ Baca dulu

1. **GPU wajib H100 atau H200** (arsitektur Hopper / compute capability `9.0` / sm90).
   RTX 4090, RTX 5090, dan A100 **ditolak** oleh binary miner resmi. Hardware Anda (H100–H200) cocok.
2. **Model default = Llama-3.3-70B** (~140 GB). H200 141GB disarankan; dengan H100 80GB
   Anda perlu **2 kartu** (set `TENSOR_PARALLEL=2`).
3. **Profitabilitas turun cepat** seiring makin banyak miner bergabung. Cek dulu
   kalkulator profit terbaru sebelum serius. Mining ini juga memakai listrik & sewa GPU.
4. Jalankan di server/GPU yang memang **Anda miliki atau sewa secara sah**. Banyak
   penyedia cloud melarang mining — pastikan Anda diizinkan.

---

## 🚀 Langkah cepat

```bash
cd pearl-miner
cp .env.example .env
nano .env                 # minimal: ganti RPC_PASS, isi HF_TOKEN
chmod +x pearl-mine.sh

./pearl-mine.sh doctor    # 1. cek GPU, driver, docker
./pearl-mine.sh install   # 2. pasang binary pearl (pearld/oyster/prlctl)
./pearl-mine.sh wallet    # 3. buat wallet + alamat mining (prl1...)
./pearl-mine.sh all       # 4. jalankan node + miner sekaligus
./pearl-mine.sh status    # cek sinkronisasi & saldo
./pearl-mine.sh logs      # pantau log
./pearl-mine.sh stop      # berhenti
```

---

## 🔑 Cara membuat ADDRESS (alamat dompet) — penjelasan detail

Di Pearl, alamat dibuat oleh **wallet daemon bernama Oyster**. Alamat Pearl
berformat **Taproot bech32m** dan selalu diawali `prl1...`.
Reward mining Anda dikirim ke alamat ini.

Cara paling mudah sudah dibungkus oleh skrip:

```bash
./pearl-mine.sh wallet
```

Tapi supaya Anda paham apa yang terjadi di balik layar, ini prosesnya manual:

### 1) Buat wallet HD baru
```bash
oyster -u rpcuser -P rpcpass --create
```
- Anda diminta membuat **passphrase** (untuk mengenkripsi private key di disk).
- Anda diberi **seed phrase** (recovery phrase). **INI PALING PENTING.**
  - Tulis di kertas, simpan offline. Jangan foto, jangan taruh di cloud.
  - Siapa pun yang punya seed = bisa mencuri semua PRL Anda.
  - Kalau GPU/server rusak, seed inilah yang memulihkan dana Anda.

### 2) Jalankan daemon wallet
```bash
oyster -u rpcuser -P rpcpass
```
Oyster kini aktif dan mendengarkan di port wallet (mainnet: `44207`).

### 3) Minta alamat mining baru
```bash
prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
```
Perintah ini mengembalikan sesuatu seperti:
```
prl1pwpd5kgw3ae7s5ewqwnl3sx0a47t8crzg87r7th7a8w23qzy6ah2qu47f3h
```
Itulah **alamat mining Anda**. Tempel ke `MINING_ADDRESS=` di file `.env`
(skrip `wallet` melakukannya otomatis).

> Alternatif interaktif: jalankan `oystercli` lalu pilih menu **Receive** untuk
> membuat wallet & alamat lewat panduan langkah demi langkah.

### Cara aman menyimpan
- **Seed phrase** → kertas / metal backup, offline. Ini kunci utama.
- **Passphrase** → password manager. Tanpa ini wallet lokal tak bisa dibuka.
- Untuk menampung reward besar dalam jangka panjang, pertimbangkan memindahkannya
  ke wallet terpisah yang tidak ada di server mining.

---

## 🧩 Menjalankan komponen satu per satu

| Perintah | Fungsi |
|----------|--------|
| `./pearl-mine.sh doctor`  | Cek GPU sm90, driver, Docker, NVIDIA runtime, binary |
| `./pearl-mine.sh install` | Unduh & pasang `pearld`, `oyster`, `prlctl`, `oystercli` |
| `./pearl-mine.sh wallet`  | Buat wallet + generate alamat `prl1...` (disimpan ke `.env`) |
| `./pearl-mine.sh node`    | Jalankan full node `pearld` (sinkronisasi + mining coordinator) |
| `./pearl-mine.sh miner`   | Jalankan vLLM miner (GPU) via Docker |
| `./pearl-mine.sh all`     | node + miner sekaligus |
| `./pearl-mine.sh status`  | Info blockchain + saldo + status container |
| `./pearl-mine.sh logs`    | Tail log node & wallet |
| `./pearl-mine.sh stop`    | Hentikan semua |

---

## 🐳 Menjalankan miner TANPA skrip (Docker manual)

```bash
docker run --rm -it --gpus all --network host --shm-size 8g \
  -e PEARLD_RPC_URL=http://localhost:44107 \
  -e PEARLD_RPC_USER=rpcuser \
  -e PEARLD_RPC_PASSWORD=rpcpass \
  -e HF_TOKEN=hf_xxx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm_miner:latest \
  pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

## 🛠️ Menjalankan miner TANPA Docker (dari source)

Butuh Python 3.12 + [uv](https://docs.astral.sh/uv/) + CUDA toolkit:

```bash
git clone https://github.com/pearl-research-labs/pearl && cd pearl
uv sync --package vllm-miner        # compile kernel pearl-gemm (sm90) otomatis

export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="rpcuser"
export PEARLD_RPC_PASSWORD="rpcpass"
export PEARLD_MINING_ADDRESS="prl1...."   # alamat Anda
pearl-gateway start                  # jembatan node <-> miner
# lalu jalankan vllm serve dengan plugin pearl (lihat miner/README.md)
```

---

## ❓ Troubleshooting singkat

- **`compute capability` bukan 9.0** → GPU tidak didukung. Wajib H100/H200.
- **Model gagal diunduh** → isi `HF_TOKEN` di `.env` (buat di huggingface.co/settings/tokens)
  dan setujui lisensi model di halaman HF-nya.
- **Node lama sinkron** → normal, tunggu sampai `getblockchaininfo` menunjukkan sinkron.
- **VRAM kurang (OOM)** → turunkan `--max-model-len`, atau pakai `TENSOR_PARALLEL=2` dengan 2 GPU.

## 📚 Referensi
- Repo resmi: https://github.com/pearl-research-labs/pearl
- Situs: https://pearlresearch.ai
- Paper PoUW: https://arxiv.org/abs/2504.09971
