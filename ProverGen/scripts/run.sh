SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NUM=2000

export NOISE1=1
export NOISE2=1
export CNOISE1=0
export CNOISE2=0

bash easy.sh
bash medium.sh
bash hard.sh
