#!/usr/bin/env bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

main() {
    log_info "Starting lacuna installation..."

    if ! command -v git &> /dev/null; then
        log_info "Installing git..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git
        else
            log_warn "Please install git manually"
            exit 1
        fi
    fi

    log_info "Cloning lacuna repository..."
    git clone https://github.com/keatonelvins/lacuna.git
    cd lacuna

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # TODO: enforce uv version

    log_info "Setting up uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Installing Python dependencies..."
    uv sync --dev

    log_info "Installation complete!"
    log_info "See options with uv run pt --help or uv run sft --help"
}

main