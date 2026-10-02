#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
source_file="hartree_treecode_ks_scf_beamer_en.tex"
format_root="/tmp/hartree-treecode-xelatex-format"
format_dir="$format_root/web2c/xetex"

cd "$script_dir"

if command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error "$source_file"
  xelatex -interaction=nonstopmode -halt-on-error "$source_file"
else
  mkdir -p "$format_dir"
  if [[ ! -f "$format_dir/xelatex.fmt" ]]; then
    (
      cd "$format_dir"
      env TEXMFVAR="$format_root" \
        xetex -ini -jobname=xelatex -progname=xelatex -etex xelatex.ini
    )
  fi

  env TEXMFVAR="$format_root" \
    xetex -fmt=xelatex -progname=xelatex \
      -interaction=nonstopmode -halt-on-error "$source_file"
  env TEXMFVAR="$format_root" \
    xetex -fmt=xelatex -progname=xelatex \
      -interaction=nonstopmode -halt-on-error "$source_file"
fi

echo "Built $script_dir/hartree_treecode_ks_scf_beamer_en.pdf"
