"""
Parser untuk file hasil inversi RES2DINV (.INV)
Mendukung format output RES2DINV versi 3.x - 10.x

Struktur file .INV RES2DINV:
  Baris 1        : Nama lintasan
  Baris 2        : Spasi elektroda
  Baris 3        : Tipe array (kode numerik)
  Baris 4        : Jumlah datum
  Baris 5        : 1 (flag)
  Baris 6        : 0 (flag)
  Baris 7..N     : Data apparent resistivity terukur
                   format: x_mid, a, n, rho_a
  ---
  INVERSION RESULTS
  Initial RMS error
  <nilai>
  NUMBER OF LAYERS
  <n_layer>
  NUMBER OF BLOCKS
  <n_blok>
  ITERATION <k>
  MODEL RESISTIVITY
  LAYER <l>
  <n_blok_layer>, <kedalaman_layer>
      <x_pos>,   <rho>
      ...
      <whitespace><rho_sisi>   (baris terakhir layer, hanya satu nilai)
  ...
  CALCULATED APPARENT RESISTIVITY
      x, a, n, rho_calc
  <rms_value>
"""

import numpy as np
import re


# ─── Parser utama ──────────────────────────────────────────────────────────────

def parse_inv_file(content: str) -> dict:
    """
    Parse file .INV dari RES2DINV.
    Mengembalikan dict berisi metadata, matriks resistivitas model, datum, dll.
    """
    lines = [l.rstrip() for l in content.replace("\r\n", "\n").replace("\r", "\n").splitlines()]

    result = {
        "success": False,
        "error": None,
        "metadata": {},
        "survey_name": "",
        "electrode_spacing": 1.0,
        "array_type": 7,
        "array_name": "Wenner-Schlumberger",
        "n_datum": 0,
        "n_electrodes": 0,
        # Measured apparent resistivity (datum pseudosection)
        "measured_datum": None,     # np.array (N,3): [x_mid, pseudo_depth, rho_a]
        # Calculated apparent resistivity (iterasi terpilih)
        "calculated_datum": None,   # np.array (N,3): [x_mid, pseudo_depth, rho_a]
        # Model resistivity
        "rho_matrix": None,         # np.array (n_layers, n_cols_max)
        "depths": [],               # kedalaman tiap layer (m)
        "x_positions": [],          # posisi x kolom model (m)
        "n_rows": 0,
        "n_cols": 0,
        "rms_history": [],
        "final_rms": None,
        "iteration_used": 0,
        "n_iterations": 0,
        "initial_rms": None,
        "iterations": [],           # list detail tiap iterasi
    }

    try:
        idx = 0
        n = len(lines)

        # ── Baris 1: nama lintasan ──────────────────────────────────────────
        result["survey_name"] = lines[0].strip()
        idx = 1

        # ── Baris 2: spasi elektroda ────────────────────────────────────────
        m = re.search(r"([\d.]+)", lines[idx])
        if m:
            result["electrode_spacing"] = float(m.group(1))
        idx += 1

        # ── Baris 3: kode array ─────────────────────────────────────────────
        m = re.search(r"(\d+)", lines[idx])
        if m:
            result["array_type"] = int(m.group(1))
        result["array_name"] = _array_name(result["array_type"])
        idx += 1

        # ── Baris 4: jumlah datum ───────────────────────────────────────────
        m = re.search(r"(\d+)", lines[idx])
        if m:
            result["n_datum"] = int(m.group(1))
        idx += 1

        # ── Baris 5 & 6: flag (1 dan 0) ────────────────────────────────────
        idx += 2   # lewati dua baris flag

        # ── Baris datum apparent resistivity terukur ────────────────────────
        measured = []
        a = result["electrode_spacing"]
        while idx < n:
            ln = lines[idx].strip()
            vals = _parse_floats(ln)
            if len(vals) >= 4:
                x_mid, a_val, n_val, rho_a = vals[0], vals[1], vals[2], vals[3]
                # pseudo-depth dari spasi a dan level n
                z = _pseudo_depth(result["array_type"], a_val, n_val)
                measured.append([x_mid, z, rho_a])
                idx += 1
            else:
                break

        if measured:
            result["measured_datum"] = np.array(measured)
            # Perkiraan jumlah elektroda dari jangkauan x
            x_arr = result["measured_datum"][:, 0]
            result["n_electrodes"] = int(round(
                (x_arr.max() - x_arr.min()) / a)) + 3

        # ── Cari blok "INVERSION RESULTS" ───────────────────────────────────
        while idx < n and "INVERSION RESULTS" not in lines[idx].upper():
            idx += 1
        idx += 1  # lewati baris "INVERSION RESULTS"

        # ── Initial RMS ──────────────────────────────────────────────────────
        while idx < n and "INITIAL RMS" not in lines[idx].upper():
            idx += 1
        idx += 1
        m = re.search(r"([\d.]+)", lines[idx])
        if m:
            result["initial_rms"] = float(m.group(1))
        idx += 1

        # ── NUMBER OF LAYERS ─────────────────────────────────────────────────
        while idx < n and "NUMBER OF LAYERS" not in lines[idx].upper():
            idx += 1
        idx += 1
        m = re.search(r"(\d+)", lines[idx])
        n_layers = int(m.group(1)) if m else 0
        idx += 1

        # ── NUMBER OF BLOCKS ─────────────────────────────────────────────────
        while idx < n and "NUMBER OF BLOCKS" not in lines[idx].upper():
            idx += 1
        idx += 1
        idx += 1  # lewati nilai

        # ── Baca semua iterasi ───────────────────────────────────────────────
        iterations = []

        while idx < n:
            # Cari "ITERATION k"
            if re.search(r"^ITERATION\s+\d+", lines[idx].strip(), re.IGNORECASE):
                iter_num_m = re.search(r"ITERATION\s+(\d+)", lines[idx], re.IGNORECASE)
                iter_num = int(iter_num_m.group(1)) if iter_num_m else len(iterations) + 1
                idx += 1

                # "MODEL RESISTIVITY"
                while idx < n and "MODEL RESISTIVITY" not in lines[idx].upper():
                    idx += 1
                idx += 1

                # Baca semua layer
                layers = {}
                layer_depths = {}

                while idx < n:
                    ln = lines[idx].strip()

                    # "LAYER l"
                    lm = re.match(r"LAYER\s+(\d+)", ln, re.IGNORECASE)
                    if lm:
                        layer_idx = int(lm.group(1))
                        idx += 1

                        # Baris berikut: "n_blok, depth"
                        ln2 = lines[idx].strip()
                        depth_m = re.search(r"[\d.]+", ln2.split(",")[-1]) if "," in ln2 else None
                        if depth_m:
                            layer_depth = float(depth_m.group())
                        else:
                            # Layer 8 sering tidak punya kedalaman
                            layer_depth = None
                        idx += 1

                        # Baca nilai resistivitas blok-blok di layer ini
                        layer_data = []   # list of (x_pos, rho)
                        last_rho = None   # nilai sisi terakhir (tanpa x_pos)

                        while idx < n:
                            ln3 = lines[idx].strip()
                            # Cek apakah ini baris layer baru / section baru
                            if (re.match(r"LAYER\s+\d+", ln3, re.IGNORECASE) or
                                    re.match(r"CALCULATED APPARENT", ln3, re.IGNORECASE) or
                                    re.match(r"ITERATION\s+\d+", ln3, re.IGNORECASE) or
                                    re.match(r"REFERENCE MODEL", ln3, re.IGNORECASE)):
                                break

                            vals = _parse_floats(ln3)
                            if len(vals) == 2:
                                # x_pos, rho
                                layer_data.append((vals[0], vals[1]))
                                idx += 1
                            elif len(vals) == 1 and ln3.startswith(" " * 15):
                                # Nilai sisi (boundary) terakhir tanpa x_pos
                                last_rho = vals[0]
                                idx += 1
                            elif len(vals) == 0:
                                idx += 1
                            else:
                                # Bisa jadi nilai sisi yang lebih pendek indentasinya
                                # coba cek apakah baris ini diindent lebih dari biasa
                                if ln3 and not ln3[0].isdigit() and len(vals) == 1:
                                    last_rho = vals[0]
                                    idx += 1
                                else:
                                    break

                        layers[layer_idx] = layer_data
                        layer_depths[layer_idx] = layer_depth
                        continue

                    # "CALCULATED APPARENT RESISTIVITY"
                    if "CALCULATED APPARENT" in ln.upper():
                        idx += 1
                        calc = []
                        a_sp = result["electrode_spacing"]
                        while idx < n:
                            ln4 = lines[idx].strip()
                            # Baris RMS (satu angka saja di baris baru)
                            if re.match(r"^\s*[\d.]+\s*$", lines[idx]) and len(_parse_floats(ln4)) == 1:
                                rms_val = _parse_floats(ln4)[0]
                                idx += 1
                                break
                            vals4 = _parse_floats(ln4)
                            if len(vals4) >= 4:
                                x_mid, a_val, n_val, rho_c = vals4[0], vals4[1], vals4[2], vals4[3]
                                z = _pseudo_depth(result["array_type"], a_val, n_val)
                                calc.append([x_mid, z, rho_c])
                                idx += 1
                            elif len(vals4) == 1:
                                # Ini adalah nilai RMS di akhir section
                                rms_val = vals4[0]
                                idx += 1
                                break
                            else:
                                idx += 1
                                break

                        iterations.append({
                            "iter": iter_num,
                            "rms": rms_val if "rms_val" in dir() else 99.0,
                            "layers": layers,
                            "layer_depths": layer_depths,
                            "calculated_datum": np.array(calc) if calc else None,
                        })
                        rms_val_clean = iterations[-1]["rms"]
                        result["rms_history"].append(rms_val_clean)
                        break  # setelah CALCULATED APPARENT lanjut ke iterasi berikutnya

                    idx += 1

            else:
                idx += 1

        # ── Pilih iterasi terbaik (RMS terkecil) ────────────────────────────
        result["iterations"] = iterations
        result["n_iterations"] = len(iterations)

        if not iterations:
            result["error"] = "Tidak ada data iterasi yang berhasil diparsing."
            return result

        best = min(iterations, key=lambda it: it["rms"])
        result["final_rms"] = best["rms"]
        result["iteration_used"] = best["iter"]
        result["calculated_datum"] = best["calculated_datum"]

        # Semua rms history dari tabel di akhir file (lebih lengkap)
        # Kalau sudah terisi dari parsing iterasi, pakai itu
        if not result["rms_history"] and result["initial_rms"]:
            result["rms_history"] = [result["initial_rms"]]

        # ── Bangun matriks resistivitas dari iterasi terpilih ───────────────
        _build_rho_matrix(best, result)

        result["success"] = result["rho_matrix"] is not None
        if not result["success"]:
            result["error"] = "Matriks resistivitas tidak dapat dibangun dari data layer."

        return result

    except Exception as e:
        import traceback
        result["error"] = f"Error parsing file: {str(e)}\n{traceback.format_exc()}"
        return result


# ─── Bangun matriks 2D dari data layer ────────────────────────────────────────

def _build_rho_matrix(iteration: dict, result: dict):
    """
    Susun matriks 2D [n_layers × n_cols] dari data per-layer.
    Tiap layer punya list (x_pos, rho) dengan jumlah kolom berbeda (menyempit ke dalam).
    """
    layers = iteration["layers"]
    layer_depths_raw = iteration["layer_depths"]
    if not layers:
        return

    n_layers = max(layers.keys())

    # Kumpulkan semua x_pos unik dari semua layer
    all_x = set()
    for ldata in layers.values():
        for (xp, rho) in ldata:
            all_x.add(round(xp, 3))
    all_x = sorted(all_x)

    if not all_x:
        return

    n_cols = len(all_x)
    x_map = {x: i for i, x in enumerate(all_x)}

    # Kedalaman tiap layer
    depths = []
    for li in range(1, n_layers + 1):
        d = layer_depths_raw.get(li)
        if d is not None:
            depths.append(d)
        elif depths:
            # Perkiraan dari layer sebelumnya × 1.1
            depths.append(round(depths[-1] * 1.1, 4))
        else:
            depths.append(result["electrode_spacing"] * 0.5)

    # Bangun matriks
    mat = np.full((n_layers, n_cols), np.nan)
    for li in range(1, n_layers + 1):
        ldata = layers.get(li, [])
        for (xp, rho) in ldata:
            xi = x_map.get(round(xp, 3))
            if xi is not None and rho > 0:
                mat[li - 1, xi] = rho

    result["rho_matrix"] = mat
    result["depths"] = depths
    result["x_positions"] = all_x
    result["n_rows"] = n_layers
    result["n_cols"] = n_cols

    # Rata-rata per lapisan
    layer_avgs = []
    for ri, depth in enumerate(depths):
        row = mat[ri]
        valid = row[~np.isnan(row)]
        valid = valid[valid > 0]
        avg = float(np.mean(valid)) if len(valid) > 0 else 0.0
        layer_avgs.append({
            "layer": ri + 1,
            "depth": depth,
            "avg_rho": round(avg, 2),
            "min_rho": round(float(valid.min()), 2) if len(valid) > 0 else 0.0,
            "max_rho": round(float(valid.max()), 2) if len(valid) > 0 else 0.0,
        })
    result["layer_averages"] = layer_avgs


# ─── Helper ────────────────────────────────────────────────────────────────────

def _parse_floats(s: str) -> list:
    """Ambil semua angka float dari sebuah string."""
    return [float(x) for x in re.findall(r"-?[\d]+(?:\.[\d]+)?(?:[eE][+-]?\d+)?", s)]


def _pseudo_depth(array_code: int, a: float, n: float) -> float:
    """Hitung pseudo-depth dari spasi a dan level n."""
    if array_code == 1:    # Wenner
        return 0.500 * a
    elif array_code == 3:  # Dipole-Dipole
        return 0.300 * n * a
    else:                  # Wenner-Schlumberger (default)
        return 0.519 * n * a


def _array_name(code: int) -> str:
    return {1: "Wenner", 3: "Dipole-Dipole", 7: "Wenner-Schlumberger"}.get(
        code, "Wenner-Schlumberger"
    )


def get_array_name(code: int) -> str:
    return _array_name(code)


# ─── Demo INV generator ────────────────────────────────────────────────────────

def generate_demo_inv() -> str:
    """
    Generate contoh file .INV sintetik untuk demo/testing.
    Format identik dengan output RES2DINV asli.
    """
    np.random.seed(42)
    name = "Demo_Lintasan_WS"
    a = 1.0
    array_code = 7
    n_elec = 24
    n_layers = 6
    n_max = 8

    # Kedalaman tiap layer (WS: depth_factor * n * a)
    layer_depths = [round(0.519 * (i + 1) * a, 4) for i in range(n_layers)]

    # Model resistivitas sintetik
    rho_model = [
        [8, 9, 10, 10, 11, 12, 15, 18, 20, 25, 30, 35, 42, 50, 55, 60, 65, 70, 72, 75, 70],
        [7, 8, 9, 10, 12, 14, 18, 22, 28, 35, 45, 55, 65, 75, 80, 85, 90, 95, 90, 85],
        [6, 7, 9, 11, 14, 18, 24, 32, 42, 55, 70, 90, 110, 130, 140, 145, 140, 135],
        [6, 7, 9, 12, 16, 22, 32, 45, 60, 80, 105, 130, 155, 165, 165, 155],
        [7, 9, 12, 17, 24, 36, 52, 72, 98, 130, 162, 185, 195],
        [9, 13, 20, 32, 50, 78, 115, 165, 230, 310],
    ]

    lines = [name, f"    {a:.6f}", f"{array_code}"]

    # Hitung jumlah datum
    total_datum = sum(n_elec - (2 * n + 1) for n in range(1, n_max + 1)
                      if n_elec - (2 * n + 1) > 0)
    lines.append(str(total_datum))
    lines.append("1")
    lines.append("0")

    # Data apparent resistivity terukur (sintetik)
    for n in range(1, n_max + 1):
        n_pts = n_elec - (2 * n + 1)
        if n_pts <= 0:
            break
        for i in range(n_pts):
            x_mid = (i + n + 0.5) * a
            rho_a = 10 * n * (1 + 0.1 * np.random.randn())
            lines.append(f"    {x_mid:.6f},     {a:.6f},   {float(n):.5f},      {rho_a:.4f}")

    lines.append("INVERSION RESULTS")
    lines.append("Initial RMS error")
    lines.append(" 45.00")
    lines.append("NUMBER OF LAYERS")
    lines.append(str(n_layers))
    lines.append("NUMBER OF BLOCKS")
    lines.append(str(sum(len(r) for r in rho_model)))

    rms_seq = [45.0, 25.0, 15.0, 10.0]
    for it, rms in enumerate(rms_seq, 1):
        lines.append(f"ITERATION {it}")
        lines.append("MODEL RESISTIVITY")
        for li, (depth, row) in enumerate(zip(layer_depths, rho_model), 1):
            n_blok = len(row) - 1
            lines.append(f"LAYER {li}")
            lines.append(f"{n_blok},     {depth:.6f}")
            for ci, rho in enumerate(row[:-1]):
                x_pos = (ci + 2) * a
                noise = 1 + 0.05 * np.random.randn()
                lines.append(f"    {x_pos:.6f},      {rho * noise:.4f}")
            lines.append(f"                   {row[-1]:.4f}")
        lines.append("CALCULATED APPARENT RESISTIVITY")
        for n in range(1, n_max + 1):
            n_pts = n_elec - (2 * n + 1)
            if n_pts <= 0:
                break
            for i in range(n_pts):
                x_mid = (i + n + 0.5) * a
                rho_c = 10 * n * (1 + 0.05 * np.random.randn())
                lines.append(f"    {x_mid:.6f},     {a:.6f},   {float(n):.5f},      {rho_c:.4f}")
        lines.append(f" {rms:.3f}")

    return "\n".join(lines)
