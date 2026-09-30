#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/pdf"
mkdir -p "$OUT"

fetch () {
  local name="$1"
  local url="$2"
  echo "==> $name"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$OUT/$name" "$url"
  else
    wget -O "$OUT/$name" "$url"
  fi
}

fetch "01_norton_2015_projection_discrete_gradient.pdf"   "https://arxiv.org/pdf/1302.2713"
fetch "02_ketcheson_2019_relaxation_rk_norms.pdf"   "https://arxiv.org/pdf/1905.09847"
fetch "03_ranocha_ketcheson_2020_hamiltonian_rrk.pdf"   "https://arxiv.org/pdf/2001.04826"
fetch "04_ranocha_et_al_2020_entropy_relaxation_rk.pdf"   "https://arxiv.org/pdf/1905.09129"
fetch "05_ishii_sato_matsuo_2024_affine_projection.pdf"   "https://www.jstage.jst.go.jp/article/jsiaml/16/0/16_49/_pdf"
fetch "06_najafian_vermeire_2025_quasi_orthogonal_projection.pdf"   "https://arxiv.org/pdf/2409.18328"
fetch "07_cheng_liu_shen_lagrange_multiplier.pdf"   "https://arxiv.org/pdf/1911.08336"
fetch "08_li_et_al_2026_damped_hamiltonian_sei.pdf"   "https://arxiv.org/pdf/2603.01709"
fetch "09_tapley_2025_homogeneous_projection.pdf"   "https://arxiv.org/pdf/2511.02131"
fetch "10_tzounas_dassios_milano_2022_tdi_modes.pdf"   "https://arxiv.org/pdf/2201.09529"
fetch "11_huang_sun_2022_energy_power_system.pdf"   "https://curent.utk.edu/wp-content/uploads/2024/07/Kaiyang_Huang_UTK_Kai_1_R0.pdf"

echo
echo "Downloaded papers to $OUT"
echo "Review papers/math_projection_literature.md before using them in the manuscript."
