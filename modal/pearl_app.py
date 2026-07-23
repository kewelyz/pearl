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
