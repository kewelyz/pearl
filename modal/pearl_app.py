"""
pearl_app.py  --  Menjalankan Pearl (PRL) di Modal dengan GPU H200.

Kenapa lewat Modal Function (bukan langsung di notebook)?
  Notebook Modal default = Debian 12 (glibc 2.36). Binary resmi Pearl butuh
  glibc 2.39. Function ini memakai image Ubuntu 24.04 (glibc 2.39) sehingga
  binary pearld/oyster/prlctl langsung jalan.

Data wallet + blockchain disimpan di Modal Volume "pearl-data" => TIDAK hilang
saat container mati/restart (penting: di sinilah wallet Anda tersimpan).

Pemakaian (jalankan dari cell notebook pakai '!', atau dari terminal lokal):
  1) Buat wallet + address:
       modal run pearl_app.py::wallet --passphrase "PASSWORD_KUAT_ANDA"
     -> mencetak SEED PHRASE (CATAT DI KERTAS) dan ADDRESS (prl1...)

  2) (nanti) Jalankan mining -- lihat fungsi mine() di bawah.
"""

import modal

# ---- Konfigurasi ----
PEARL_VER = "v1.1.6"          # versi node (tag vX.Y.Z). Ganti bila perlu.
DATA = "/data"                # mount point volume persisten
APPDATA = f"{DATA}/oyster"    # data wallet (db, cert) -> persisten
NODEDATA = f"{DATA}/pearld"   # data blockchain -> persisten
RPC_USER = "rpcuser"
RPC_PASS = "pearl-rpc-pass"   # kredensial RPC lokal (boleh diganti)

# Volume persisten untuk wallet + chain + cache model
vol = modal.Volume.from_name("pearl-data", create_if_missing=True)

# Image Ubuntu 24.04 (glibc 2.39) + binary Pearl sudah dipasang di dalam image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "ca-certificates")
    .pip_install("pexpect")
    .run_commands(
        "curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/"
        f"master/install.sh | sh -s -- --version {PEARL_VER}"
    )
    .env({"PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}"})
)

app = modal.App("pearl", image=image)


@app.function(volumes={DATA: vol}, timeout=1800)
def create_wallet(passphrase: str):
    """Buat wallet baru (SPV) di volume, lalu kembalikan seed phrase + address."""
    import os, time, subprocess, pexpect

    os.makedirs(APPDATA, exist_ok=True)

    # 1) Buat wallet secara "interaktif" via pexpect (prompt butuh PTY).
    cmd = (f"oyster --appdata={APPDATA} --usespv "
           f"-u {RPC_USER} -P {RPC_PASS} --create")
    c = pexpect.spawn(cmd, encoding="utf-8", timeout=600)
    c.expect("passphrase for your new wallet"); c.sendline(passphrase)
    c.expect("Confirm passphrase");             c.sendline(passphrase)
    c.expect("additional layer of encryption"); c.sendline("no")
    c.expect("existing wallet seed");           c.sendline("no")
    c.expect('enter "OK" to continue');         raw = c.before
    c.sendline("OK"); c.expect(pexpect.EOF)

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    mnemonic = next((lines[i + 1] for i, l in enumerate(lines)
                     if "seed phrase is" in l.lower() and i + 1 < len(lines)), None)

    # 2) Jalankan daemon wallet (SPV) sebentar untuk minta address.
    daemon = subprocess.Popen(
        ["oyster", f"--appdata={APPDATA}", "--usespv",
         "-u", RPC_USER, "-P", RPC_PASS],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    address = None
    try:
        for _ in range(40):
            time.sleep(3)
            r = subprocess.run(
                ["prlctl", "--wallet", "-u", RPC_USER, "-P", RPC_PASS,
                 "-s", "127.0.0.1:44207", "-c", f"{APPDATA}/rpc.cert",
                 "getnewaddress"],
                capture_output=True, text=True,
            )
            out = (r.stdout or "").strip().strip('"')
            if out.startswith("prl1"):
                address = out
                break
    finally:
        daemon.terminate()

    vol.commit()  # pastikan wallet tersimpan permanen di volume
    return {"mnemonic": mnemonic, "address": address}


@app.local_entrypoint()
def wallet(passphrase: str = "ganti-passphrase-anda"):
    res = create_wallet.remote(passphrase)
    print("=" * 64)
    print("SEED PHRASE (CATAT DI KERTAS, JANGAN dibagikan / screenshot):")
    print("   ", res["mnemonic"])
    print("-" * 64)
    print("ADDRESS MINING (boleh dibagikan, ini alamat penerima reward):")
    print("   ", res["address"])
    print("=" * 64)


# ============================================================================
#  MINING
# ============================================================================
MODEL = "pearl-ai/Llama-3.3-70B-Instruct-pearl"

# Image miner: CUDA 13.0.3 devel (Ubuntu 24.04) + rust + uv + node + miner build.
# Mengikuti Dockerfile & miner/README.md resmi ("uv sync --package vllm-miner").
miner_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "git", "build-essential", "ca-certificates",
                 "pkg-config", "clang", "llvm", "python3-dev")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/"
        f"master/install.sh | sh -s -- --version {PEARL_VER}",
        "git clone --depth 1 https://github.com/pearl-research-labs/pearl /opt/pearl",
    )
    .env({
        "PATH": "/root/.local/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH}",
        "UV_TORCH_BACKEND": "cu130",
        "CARGO_HOME": "/root/.cargo",
        "RUSTUP_HOME": "/root/.rustup",
    })
    # Build miner (torch + vLLM + kompilasi kernel pearl-gemm sm90). Lama (~20-40 mnt).
    .run_commands("cd /opt/pearl && uv sync --package vllm-miner")
)


@app.function(
    image=miner_image,
    gpu="H200",
    volumes={DATA: vol},
    timeout=60 * 60 * 24,                       # maksimum Modal 24 jam; jalankan ulang untuk lanjut
    secrets=[modal.Secret.from_name("huggingface")],  # berisi HF_TOKEN
)
def mine(address: str):
    """Jalankan node pearld + gateway + vLLM miner di 1 GPU H200."""
    import os, subprocess, time

    os.makedirs(NODEDATA, exist_ok=True)
    os.makedirs(f"{DATA}/hf", exist_ok=True)

    env = {
        **os.environ,
        "PEARLD_RPC_URL": "http://127.0.0.1:44107",
        "PEARLD_RPC_USER": RPC_USER,
        "PEARLD_RPC_PASSWORD": RPC_PASS,
        "PEARLD_MINING_ADDRESS": address,
        "HF_HOME": f"{DATA}/hf",                 # cache model 70B -> persisten di volume
    }

    # 1) Node pearld (RPC tanpa TLS -> gateway konek via http), data di volume.
    print(">> start pearld ...")
    subprocess.Popen(
        ["pearld", "--notls", "--rpclisten=127.0.0.1:44107",
         "-u", RPC_USER, "-P", RPC_PASS,
         f"--datadir={NODEDATA}", f"--miningaddr={address}", "--txindex"],
    )
    time.sleep(15)

    # 2) pearl-gateway (jembatan node <-> miner), buat socket /tmp/pearlgw.sock.
    print(">> start pearl-gateway ...")
    subprocess.Popen(["uv", "run", "pearl-gateway", "start"], cwd="/opt/pearl", env=env)
    for _ in range(60):
        if os.path.exists("/tmp/pearlgw.sock"):
            break
        time.sleep(2)

    # 3) vLLM serve dengan plugin Pearl (unduh model ~140GB pertama kali).
    print(">> start vllm serve (mining) ...")
    subprocess.run(
        ["uv", "run", "vllm", "serve", MODEL,
         "--host", "0.0.0.0", "--port", "8000",
         "--max-model-len", "8192",
         "--gpu-memory-utilization", "0.9",
         "--enforce-eager"],
        cwd="/opt/pearl", env=env,
    )


@app.local_entrypoint()
def start_mining(address: str):
    """modal run pearl_app.py::start_mining --address prl1...."""
    mine.remote(address)
