#!/usr/bin/env bash
# One-time infra setup for this WSL2 environment: Docker Engine, kubectl, minikube.
# Must be run in a REAL terminal (not via Claude Code) since it needs an interactive
# sudo password prompt this session has no TTY to answer.
#
# Usage: bash scripts/install_infra.sh
set -euo pipefail

echo "=== Docker Engine ==="
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "=== kubectl ==="
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
rm -f kubectl.sha256 2>/dev/null || true

echo "=== minikube ==="
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube-linux-amd64 && sudo mv minikube-linux-amd64 /usr/local/bin/minikube

echo
echo "=== Done. Next steps (required): ==="
echo "1. From WINDOWS PowerShell (not this WSL shell): wsl.exe --shutdown"
echo "2. Reopen your WSL terminal (refreshes docker group membership)"
echo "3. Verify: docker run hello-world && kubectl version --client && minikube version"
