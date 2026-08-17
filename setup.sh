#!/bin/bash
set -e

###############################################
# Globals
###############################################
REPO_DIR="$(pwd)"              # Install where the repo was cloned
VENV_DIR="$REPO_DIR/.venv"

PACKAGES=("git" "python3" "python3-dev" "python3-pip" "python3-venv" "wget" "curl")
TO_INSTALL=()
KEYRING_URL=""
APT_GET=""

###############################################
# Detect package manager (apt-get or apt)
###############################################
detect_apt() {
    if command -v apt-get &>/dev/null; then
        APT_GET="apt-get"
    elif command -v apt &>/dev/null; then
        APT_GET="apt"
    else
        echo "Error: Neither apt-get nor apt found. This script requires a Debian-based system."
        exit 1
    fi
}

###############################################
# Detect WSL
###############################################
is_wsl() {
    grep -qi "microsoft" /proc/version || [[ -n "$WSL_DISTRO_NAME" ]]
}

###############################################
# Detect NVIDIA GPU
###############################################
has_nvidia_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi &>/dev/null && return 0
    fi
    return 1
}

###############################################
# Distro detection
###############################################
get_distro_settings() {
    if is_wsl; then
        echo "Detected WSL environment"
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
        return 0
    fi

    if [ -f /etc/os-release ]; then
        . /etc/os-release
    else
        echo "Error: Cannot detect OS"
        return 1
    fi

    case "$ID" in
        ubuntu)
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        debian)
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
            ;;
        *)
            echo "Unsupported distribution: $ID"
            return 1
            ;;
    esac
}

###############################################
# Install base packages
###############################################
install_base_packages() {
    for pkg in "${PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            TO_INSTALL+=("$pkg")
        fi
    done

    if [ ${#TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing packages: ${TO_INSTALL[*]}"
        sudo "$APT_GET" update
        sudo "$APT_GET" install -y "${TO_INSTALL[@]}"
    fi
}

###############################################
# Install CUDA Toolkit
###############################################
install_cuda() {
    # Install keyring if missing
    if ! dpkg-query -W -f='${Status}' "cuda-keyring" 2>/dev/null | grep -q "install ok installed"; then
        echo "Installing CUDA keyring..."

        # Remove the old NVIDIA key if present (apt-key is deprecated/removed in newer Ubuntu)
        if command -v apt-key &>/dev/null; then
            sudo apt-key del 7fa2af80 2>/dev/null || true
        else
            # Remove legacy trusted key files directly
            sudo rm -f /etc/apt/trusted.gpg.d/cuda*.gpg 2>/dev/null || true
        fi

        wget -q "$KEYRING_URL" -O cuda-keyring.deb
        sudo dpkg -i cuda-keyring.deb
        rm cuda-keyring.deb
        sudo "$APT_GET" update
    else
        echo "CUDA keyring already installed — skipping."
    fi

    # Install CUDA toolkit (latest available version)
    if ! dpkg-query -W -f='${Status}' "cuda-toolkit" 2>/dev/null | grep -q "install ok installed"; then
        echo "Installing cuda-toolkit..."
        sudo "$APT_GET" install -y cuda-toolkit
    else
        echo "cuda-toolkit already installed — skipping."
    fi

    # Add PATH only for native Linux
    if ! is_wsl; then
        if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        fi
    fi

    echo "CUDA installation complete."
}

###############################################
# Detect installed CUDA version for PyTorch
###############################################
get_pytorch_cuda_index() {
    local cuda_ver
    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [ -f /usr/local/cuda/version.txt ]; then
        cuda_ver=$(grep -oP '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -1)
    else
        # Fallback: check the cuda-toolkit package version
        cuda_ver=$(dpkg-query -W -f='${Version}' cuda-toolkit 2>/dev/null | grep -oP '^[0-9]+\.[0-9]+')
    fi

    if [ -z "$cuda_ver" ]; then
        echo "Warning: Could not detect CUDA version, defaulting to cu126" >&2
        echo "https://download.pytorch.org/whl/cu126"
        return
    fi

    local major minor
    major="${cuda_ver%%.*}"
    minor="${cuda_ver#*.}"

    # Map to the nearest available PyTorch wheel index.
    # PyTorch publishes wheels for specific CUDA versions only:
    #   CUDA 13.x -> cu130 (stable for CUDA 13 family)
    #   CUDA 12.6-12.8 -> cu126
    #   CUDA 12.0-12.5 -> cu121
    #   CUDA 11.x -> cu118
    local cu_tag
    if [ "$major" -ge 13 ]; then
        cu_tag="cu130"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then
        cu_tag="cu126"
    elif [ "$major" -eq 12 ]; then
        cu_tag="cu121"
    else
        cu_tag="cu118"
    fi

    echo "https://download.pytorch.org/whl/${cu_tag}"
}

###############################################
# Install PyTorch (matching CUDA version)
###############################################
install_pytorch() {
    local pytorch_index
    pytorch_index=$(get_pytorch_cuda_index)
    echo "Installing PyTorch from: $pytorch_index"
    pip install torch --index-url "$pytorch_index"
}

###############################################
# Install AirLLM + dependencies
###############################################
install_airllm() {
    echo "Installing AirLLM + dependencies..."
    pip install -r "$REPO_DIR/requirements.txt"
}

###############################################
# MAIN
###############################################
echo "Installing into repo directory: $REPO_DIR"

detect_apt

if ! get_distro_settings; then
    exit 1
fi

echo "Checking for NVIDIA GPU..."
if ! has_nvidia_gpu; then
    echo "ERROR: No NVIDIA GPU detected."
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

install_base_packages
install_cuda

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

install_pytorch
install_airllm

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
