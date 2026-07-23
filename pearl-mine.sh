#!/usr/bin/env bash
# ============================================================================
#  pearl-mine.sh
#  Orchestrator untuk mining Pearl (PRL) di GPU NVIDIA H100 / H200 (sm90).
#
#  Berbasis perintah resmi dari https://github.com/pearl-research-labs/pearl
#  Komponen: pearld (node) + oyster (wallet) + prlctl (RPC) + vLLM miner (GPU).
#
#  Pemakaian:
#     ./pearl-mine.sh doctor     # cek prasyarat (GPU, driver, docker, dll)
#     ./pearl-mine.sh install    # pasang binary pearl (pearld/oyster/prlctl)
#     ./pearl-mine.sh build      # build vLLM miner dari source (tanpa Docker, pakai uv)
#     ./pearl-mine.sh wallet     # buat wallet + generate alamat mining (prl1...)
#     ./pearl-mine.sh node       # jalankan full node (pearld)
#     ./pearl-mine.sh miner      # jalankan vLLM miner (GPU) via Docker
#     ./pearl-mine.sh all        # node + miner sekaligus
#     ./pearl-mine.sh status     # lihat status & sinkronisasi
#     ./pearl-mine.sh stop       # hentikan node & miner
#     ./pearl-mine.sh logs       # tail log node & gateway
# ============================================================================

set -Eeuo pipefail

# ---------- lokasi & konfigurasi ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# ---------- warna ----------
if [[ -t 1 ]]; then
  C_RESET='\033[0m'; C_RED='\033[31m'; C_GRN='\033[32m'
  C_YEL='\033[33m'; C_BLU='\033[34m'; C_BOLD='\033[1m'
else
  C_RESET=''; C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_BOLD=''
fi
info()  { echo -e "${C_BLU}[i]${C_RESET} $*"; }
ok()    { echo -e "${C_GRN}[ok]${C_RESET} $*"; }
warn()  { echo -e "${C_YEL}[!]${C_RESET} $*"; }
err()   { echo -e "${C_RED}[x]${C_RESET} $*" >&2; }
die()   { err "$*"; exit 1; }
hr()    { echo -e "${C_BOLD}------------------------------------------------------------${C_RESET}"; }

# ---------- muat .env ----------
load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a; # shellcheck disable=SC1090
    source "$ENV_FILE"; set +a
  else
    warn "File .env belum ada. Menyalin dari .env.example ..."
    cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
    warn "Sudah dibuat: ${ENV_FILE}"
    warn "Edit dulu (minimal RPC_PASS), lalu jalankan lagi."
    exit 1
  fi
  # default aman bila kosong
  : "${RPC_USER:=rpcuser}"
  : "${RPC_PASS:?RPC_PASS wajib diisi di .env}"
  : "${NETWORK:=mainnet}"
  : "${PEARLD_RPC_HOST:=127.0.0.1}"
  : "${PEARLD_RPC_PORT:=44107}"
  : "${WALLET_RPC_PORT:=44207}"
  : "${PEARL_BIN_DIR:=$HOME/.local/bin}"
  : "${LOG_DIR:=$HOME/.pearl-miner/logs}"
  : "${MODEL:=pearl-ai/Llama-3.3-70B-Instruct-pearl}"
  : "${TENSOR_PARALLEL:=1}"
  : "${GPU_MEMORY_UTILIZATION:=0.90}"
  : "${MAX_MODEL_LEN:=8192}"
  mkdir -p "$LOG_DIR"
  export PATH="$PEARL_BIN_DIR:$PATH"

  case "$NETWORK" in
    mainnet) NET_FLAG="" ;;
    testnet) NET_FLAG="--testnet" ;;
    simnet)  NET_FLAG="--simnet" ;;
    *) die "NETWORK tidak valid: $NETWORK (pilih mainnet/testnet/simnet)" ;;
  esac
}

need() { command -v "$1" &>/dev/null; }

# ============================================================================
#  doctor  --  cek prasyarat
# ============================================================================
cmd_doctor() {
  hr; info "Memeriksa prasyarat sistem"; hr
  local fail=0

  # OS
  info "OS       : $(uname -s) $(uname -m)"

  # GPU
  if need nvidia-smi; then
    ok "nvidia-smi ditemukan"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | while IFS= read -r line; do
      echo "    GPU: $line"
    done
    # cek Hopper (H100/H200 -> compute capability 9.0)
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
    if [[ "$cc" == "9.0" ]]; then
      ok "Compute capability $cc (Hopper / H100-H200) -> DIDUKUNG"
    else
      warn "Compute capability terdeteksi: '${cc:-tidak diketahui}'."
      warn "Miner Pearl HANYA mendukung sm90 (H100/H200). GPU lain akan ditolak."
    fi
  else
    warn "nvidia-smi tidak ada (umum di container). Cek GPU via CUDA/torch ..."
    if python3 - <<'PY' 2>/dev/null
import sys
try:
    import torch
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print(f"    GPU (torch): {torch.cuda.get_device_name(0)}  cap {cc[0]}.{cc[1]}")
        sys.exit(0 if cc == (9, 0) else 3)
    sys.exit(1)
except Exception:
    sys.exit(2)
PY
    then
      ok "GPU terdeteksi via CUDA dengan compute cap 9.0 (H100/H200) -> DIDUKUNG"
    else
      case $? in
        3) warn "GPU terdeteksi via CUDA tapi BUKAN sm90. Pearl hanya mendukung H100/H200." ;;
        1) err "Tidak ada GPU CUDA aktif."; fail=1 ;;
        *) err "Tidak bisa cek GPU (torch/nvidia-smi tidak ada). Pastikan ini instance GPU."; fail=1 ;;
      esac
    fi
  fi

  # Docker (untuk vLLM miner)
  if need docker; then
    ok "docker ditemukan ($(docker --version | awk '{print $3}' | tr -d ,))"
    if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
      ok "NVIDIA Container Runtime aktif di Docker"
    else
      warn "NVIDIA runtime belum terlihat di Docker. Pasang 'nvidia-container-toolkit'."
    fi
  else
    warn "docker tidak ada. Diperlukan untuk cara termudah menjalankan vLLM miner."
  fi

  # binary pearl
  for b in pearld oyster prlctl; do
    if need "$b"; then ok "$b terpasang ($(command -v "$b"))"; else warn "$b belum terpasang -> jalankan: $0 install"; fi
  done

  hr
  [[ $fail -eq 0 ]] && ok "Pemeriksaan selesai." || die "Ada prasyarat yang belum terpenuhi."
}

# ============================================================================
#  install  --  pasang binary resmi pearl
# ============================================================================
cmd_install() {
  hr; info "Memasang binary Pearl (pearld, oyster, prlctl, oystercli)"; hr
  if ! need curl; then die "curl tidak ada. Pasang curl dulu."; fi
  info "Mengunduh installer resmi dari pearl-research-labs ..."
  curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh
  hash -r 2>/dev/null || true
  if need pearld; then
    ok "Instalasi selesai. Binary ada di: ${PEARL_BIN_DIR}"
  else
    warn "Binary belum ada di PATH. Tambahkan ke shell profile Anda:"
    echo "    export PATH=\"${PEARL_BIN_DIR}:\$PATH\""
  fi
}

# ============================================================================
#  wallet  --  buat wallet HD + generate alamat mining taproot (prl1...)
# ============================================================================
cmd_wallet() {
  hr; info "Membuat wallet & alamat mining Pearl"; hr
  need oyster  || die "oyster belum terpasang. Jalankan: $0 install"
  need prlctl  || die "prlctl belum terpasang. Jalankan: $0 install"

  local wsrv="https://127.0.0.1:${WALLET_RPC_PORT}"

  echo
  warn "PENTING: Anda akan diminta membuat PASSPHRASE dan akan diberi SEED PHRASE."
  warn "Tulis seed phrase itu di kertas/offline. Siapa pun yang punya seed = punya dana Anda."
  echo
  read -r -p "Lanjut membuat wallet baru? [y/N] " ans
  [[ "${ans,,}" == "y" ]] || { info "Dibatalkan."; return 0; }

  # 1) buat wallet (interaktif: passphrase + seed)
  info "Menjalankan pembuatan wallet ..."
  oyster -u "$RPC_USER" -P "$RPC_PASS" $NET_FLAG --create

  # 2) start daemon wallet di background
  info "Menjalankan daemon wallet (oyster) ..."
  oyster -u "$RPC_USER" -P "$RPC_PASS" $NET_FLAG >"$LOG_DIR/oyster.log" 2>&1 &
  echo $! > "$LOG_DIR/oyster.pid"
  sleep 5

  # 3) generate alamat taproot
  info "Meminta alamat mining baru (getnewaddress) ..."
  local addr
  addr="$(prlctl -u "$RPC_USER" -P "$RPC_PASS" -s "$wsrv" getnewaddress 2>>"$LOG_DIR/oyster.log" | tr -d '"[:space:]')"

  if [[ "$addr" == prl1* ]]; then
    hr
    ok "Alamat mining Anda:"
    echo -e "    ${C_BOLD}${addr}${C_RESET}"
    hr
    # simpan ke .env otomatis
    if grep -q '^MINING_ADDRESS=' "$ENV_FILE"; then
      sed -i.bak "s|^MINING_ADDRESS=.*|MINING_ADDRESS=${addr}|" "$ENV_FILE"
    else
      echo "MINING_ADDRESS=${addr}" >> "$ENV_FILE"
    fi
    ok "Alamat sudah disimpan otomatis ke ${ENV_FILE}"
  else
    err "Gagal mendapatkan alamat. Cek log: $LOG_DIR/oyster.log"
    err "Output diterima: '${addr}'"
    return 1
  fi
}

# ============================================================================
#  node  --  jalankan full node pearld
# ============================================================================
cmd_node() {
  hr; info "Menjalankan full node pearld ($NETWORK)"; hr
  need pearld || die "pearld belum terpasang. Jalankan: $0 install"
  [[ -n "${MINING_ADDRESS:-}" ]] || die "MINING_ADDRESS kosong di .env. Jalankan: $0 wallet"
  [[ "$MINING_ADDRESS" == prl1* ]] || warn "MINING_ADDRESS tidak diawali 'prl1' -- pastikan benar."

  info "Alamat mining: $MINING_ADDRESS"
  pearld \
    --rpcuser="$RPC_USER" \
    --rpcpass="$RPC_PASS" \
    --rpclisten="0.0.0.0:${PEARLD_RPC_PORT}" \
    --miningaddr="$MINING_ADDRESS" \
    --txindex \
    $NET_FLAG \
    >"$LOG_DIR/pearld.log" 2>&1 &
  echo $! > "$LOG_DIR/pearld.pid"
  sleep 3
  if kill -0 "$(cat "$LOG_DIR/pearld.pid")" 2>/dev/null; then
    ok "pearld berjalan (pid $(cat "$LOG_DIR/pearld.pid")). Log: $LOG_DIR/pearld.log"
    info "Node akan sinkronisasi blockchain dulu. Pantau: $0 logs"
  else
    err "pearld gagal start. Cek: $LOG_DIR/pearld.log"; tail -n 20 "$LOG_DIR/pearld.log" || true
    return 1
  fi
}

# ============================================================================
#  build  --  clone repo resmi + build vLLM miner dari source (tanpa Docker)
# ============================================================================
PEARL_SRC="${PEARL_SRC:-$HOME/pearl-src}"

cmd_build() {
  hr; info "Menyiapkan vLLM miner dari source (jalur tanpa Docker)"; hr
  need git || die "git tidak ada."
  need uv  || die "uv tidak ada. Pasang: curl -LsSf https://astral.sh/uv/install.sh | sh"

  # cek toolchain wajib untuk kompilasi kernel
  need nvcc  || warn "nvcc (CUDA toolkit) tidak terdeteksi -> build kernel pearl-gemm bisa gagal."
  need rustc || warn "rustc (Rust) tidak terdeteksi -> dependency py-pearl-mining bisa gagal."

  if [[ ! -d "$PEARL_SRC/.git" ]]; then
    info "Clone repo resmi ke $PEARL_SRC ..."
    git clone --depth 1 https://github.com/pearl-research-labs/pearl "$PEARL_SRC"
  else
    info "Repo resmi sudah ada di $PEARL_SRC"
  fi

  info "Kompilasi vllm-miner + kernel pearl-gemm (bisa 10-30 menit) ..."
  ( cd "$PEARL_SRC" && uv sync --package vllm-miner )
  ok "Build miner selesai."
}

# ============================================================================
#  miner  --  jalankan vLLM miner (GPU). Auto: pakai Docker jika ada,
#             kalau tidak -> jalur source (pearl-gateway + uv run vllm serve)
# ============================================================================
cmd_miner() {
  hr; info "Menjalankan vLLM miner (GPU) -- model: $MODEL"; hr
  [[ -n "${MINING_ADDRESS:-}" ]] || die "MINING_ADDRESS kosong. Jalankan: $0 wallet"
  [[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN kosong. Model mungkin gagal diunduh dari HuggingFace."
  local rpc_url="http://${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT}"

  if need docker; then
    _miner_docker "$rpc_url"
  else
    warn "Docker tidak ada -> memakai jalur source (uv)."
    _miner_source "$rpc_url"
  fi
}

_miner_docker() {
  local rpc_url="$1"
  local image="ghcr.io/pearl-research-labs/vllm_miner:latest"
  info "Menarik image miner: $image"
  if ! docker pull "$image" 2>/dev/null; then
    warn "Gagal pull image. Membangun image dari source ..."
    [[ -d "$PEARL_SRC" ]] || cmd_build
    ( cd "$PEARL_SRC" && docker buildx build -t vllm_miner:latest . -f miner/vllm-miner/Dockerfile )
    image="vllm_miner:latest"
  fi
  docker run --rm -d --name pearl-miner \
    --gpus all --network host --shm-size 8g \
    -e PEARLD_RPC_URL="$rpc_url" -e PEARLD_RPC_USER="$RPC_USER" \
    -e PEARLD_RPC_PASSWORD="$RPC_PASS" -e PEARLD_MINING_ADDRESS="$MINING_ADDRESS" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$image" "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager
  ok "Container 'pearl-miner' berjalan. Log: docker logs -f pearl-miner"
}

_miner_source() {
  local rpc_url="$1"
  [[ -d "$PEARL_SRC" ]] || { warn "Miner belum di-build."; cmd_build; }

  export PEARLD_RPC_URL="$rpc_url"
  export PEARLD_RPC_USER="$RPC_USER"
  export PEARLD_RPC_PASSWORD="$RPC_PASS"
  export PEARLD_MINING_ADDRESS="$MINING_ADDRESS"
  export HF_TOKEN="${HF_TOKEN:-}"

  # 1) pearl-gateway (jembatan node <-> miner) di background
  info "Menjalankan pearl-gateway ..."
  ( cd "$PEARL_SRC" && uv run pearl-gateway start ) >"$LOG_DIR/gateway.log" 2>&1 &
  echo $! > "$LOG_DIR/gateway.pid"
  sleep 5

  # 2) vllm serve dengan plugin pearl
  info "Menjalankan vllm serve (unduhan model ~140GB pertama kali bisa lama) ..."
  ( cd "$PEARL_SRC" && uv run vllm serve "$MODEL" \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size "$TENSOR_PARALLEL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enforce-eager ) >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! > "$LOG_DIR/vllm.pid"
  sleep 3
  ok "Miner (source) berjalan. Log gateway: $LOG_DIR/gateway.log | vllm: $LOG_DIR/vllm.log"
  info "Pantau: tail -f $LOG_DIR/vllm.log"
}

# ============================================================================
#  status
# ============================================================================
cmd_status() {
  hr; info "Status Pearl miner"; hr
  local wsrv="https://127.0.0.1:${WALLET_RPC_PORT}"
  if need prlctl; then
    info "Info blockchain (getblockchaininfo):"
    prlctl -u "$RPC_USER" -P "$RPC_PASS" $NET_FLAG getblockchaininfo 2>/dev/null || warn "Node belum siap / belum jalan."
    echo
    info "Saldo wallet (getbalance):"
    prlctl -u "$RPC_USER" -P "$RPC_PASS" -s "$wsrv" getbalance 2>/dev/null || warn "Wallet belum jalan."
  fi
  echo
  if need docker && docker ps --format '{{.Names}}' | grep -q '^pearl-miner$'; then
    ok "Container miner: BERJALAN"
  else
    warn "Container miner: tidak berjalan"
  fi
}

# ============================================================================
#  stop
# ============================================================================
cmd_stop() {
  hr; info "Menghentikan node & miner"; hr
  if need docker && docker ps --format '{{.Names}}' | grep -q '^pearl-miner$'; then
    docker stop pearl-miner >/dev/null && ok "Container miner dihentikan."
  fi
  for svc in vllm gateway pearld oyster; do
    if [[ -f "$LOG_DIR/$svc.pid" ]] && kill -0 "$(cat "$LOG_DIR/$svc.pid")" 2>/dev/null; then
      kill "$(cat "$LOG_DIR/$svc.pid")" && ok "$svc dihentikan."
      rm -f "$LOG_DIR/$svc.pid"
    fi
  done
}

# ============================================================================
#  logs
# ============================================================================
cmd_logs() {
  info "Ctrl-C untuk keluar. Menampilkan log pearld & oyster ..."
  touch "$LOG_DIR/pearld.log" "$LOG_DIR/oyster.log"
  tail -n 40 -f "$LOG_DIR/pearld.log" "$LOG_DIR/oyster.log"
}

# ============================================================================
#  all
# ============================================================================
cmd_all() {
  cmd_node
  info "Menunggu node siap 10 detik sebelum start miner ..."
  sleep 10
  cmd_miner
  cmd_status
}

usage() {
  # cetak hanya blok dokumentasi paling atas (di antara 2 garis "# ===")
  awk 'NR==1{next} /^# =====/{c++; if(c==2) exit} /^#/{sub(/^# ?/,""); print}' "$0"
}

main() {
  local sub="${1:-}"; shift || true
  case "$sub" in
    doctor)  load_env; cmd_doctor ;;
    install) load_env; cmd_install ;;
    build)   load_env; cmd_build ;;
    wallet)  load_env; cmd_wallet ;;
    node)    load_env; cmd_node ;;
    miner)   load_env; cmd_miner ;;
    all)     load_env; cmd_all ;;
    status)  load_env; cmd_status ;;
    stop)    load_env; cmd_stop ;;
    logs)    load_env; cmd_logs ;;
    ""|-h|--help|help) usage ;;
    *) err "Sub-perintah tidak dikenal: $sub"; echo; usage; exit 1 ;;
  esac
}
main "$@"
