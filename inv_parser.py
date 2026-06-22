"""
Parser untuk file hasil inversi RES2DINV (.INV)
Mendukung berbagai format output RES2DINV versi 3.x - 10.x
"""

import numpy as np
import re
import io


def parse_inv_file(content: str) -> dict:
    """
    Parse file .INV dari RES2DINV.
    Mengembalikan dict berisi metadata dan array nilai resistivitas.
    """
    lines = content.strip().splitlines()
    result = {
        "success": False,
        "error": None,
        "metadata": {},
        "blocks": [],          # list of {col, row, x_pos, depth, resistivity}
        "rho_matrix": None,    # 2D numpy array [row][col]
        "depths": [],
        "x_positions": [],
        "n_cols": 0,
        "n_rows": 0,
        "electrode_spacing": 1.0,
        "array_type": 7,       # default W-S
        "n_electrodes": 0,
        "rms_history": [],
        "final_rms": None,
        "iteration_used": 0,
    }

    try:
        # --- Cari metadata di baris awal ---
        for i, line in enumerate(lines[:30]):
            ln = line.strip().lower()
            # Spasi elektroda
            if "electrode spacing" in ln or "unit electrode spacing" in ln:
                nums = re.findall(r"[\d.]+", line)
                if nums:
                    result["electrode_spacing"] = float(nums[0])
            # Array type
            if "wenner-schlumberger" in ln or "array type" in ln and "7" in line:
                result["array_type"] = 7
            elif "wenner" in ln and "schlumberger" not in ln:
                result["array_type"] = 1
            elif "dipole" in ln:
                result["array_type"] = 3
            # Jumlah elektroda
            m = re.search(r"(\d+)\s+electrode", ln)
            if m:
                result["n_electrodes"] = int(m.group(1))

        # --- Cari semua iterasi dan ambil yang terakhir ---
        iter_blocks = []
        current_iter = None
        current_rms = None
        current_data = []
        in_data_block = False

        for line in lines:
            ln_strip = line.strip()

            # Deteksi baris iterasi
            iter_match = re.search(
                r"(?:iteration|iterasi)\s*[:\s]*(\d+).*?(?:rms|error)[^=]*=?\s*([\d.]+)\s*%",
                ln_strip, re.IGNORECASE
            )
            if not iter_match:
                iter_match = re.search(
                    r"(\d+)\s+(?:iteration|iterasi).*?([\d.]+)\s*%",
                    ln_strip, re.IGNORECASE
                )
            if not iter_match:
                iter_match = re.search(
                    r"rms[^=]*=\s*([\d.]+).*?iteration\s*(\d+)",
                    ln_strip, re.IGNORECASE
                )

            if iter_match:
                # Simpan blok sebelumnya
                if current_iter is not None and current_data:
                    iter_blocks.append({
                        "iter": current_iter,
                        "rms": current_rms,
                        "data": list(current_data)
                    })
                try:
                    g = iter_match.groups()
                    current_iter = int(g[0]) if g[0].isdigit() else int(g[1])
                    current_rms = float(g[1]) if g[0].isdigit() else float(g[0])
                except:
                    current_iter = len(iter_blocks) + 1
                    current_rms = 99.0
                current_data = []
                result["rms_history"].append(current_rms)
                in_data_block = True
                continue

            # Parse baris data numerik
            if in_data_block:
                parts = ln_strip.split()
                # Format: block_no x_pos depth resistivity  ATAU  x_pos depth resistivity
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except:
                        pass
                if len(nums) >= 3:
                    current_data.append(nums)

        # Simpan blok terakhir
        if current_iter is not None and current_data:
            iter_blocks.append({
                "iter": current_iter,
                "rms": current_rms,
                "data": list(current_data)
            })

        # Jika tidak ada blok iterasi terdeteksi, coba parse langsung
        if not iter_blocks:
            result = _parse_simple_format(lines, result)
            return result

        # Ambil iterasi terbaik (RMS terkecil)
        if iter_blocks:
            best = min(iter_blocks, key=lambda b: b["rms"])
            result["final_rms"] = best["rms"]
            result["iteration_used"] = best["iter"]
            _extract_grid(best["data"], result)

        # Fallback: hitung n_electrodes dari n_cols jika tidak terdeteksi
        if result["n_electrodes"] == 0 and result["n_cols"] > 0:
            result["n_electrodes"] = result["n_cols"] + 1

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = f"Error parsing file: {str(e)}"
        return result


def _extract_grid(raw_data: list, result: dict):
    """Ekstrak grid 2D dari data mentah."""
    if not raw_data:
        return

    # Coba deteksi format kolom
    # Format A: [block_id, x_pos, depth, rho]  (4 kolom)
    # Format B: [x_pos, depth, rho]             (3 kolom)
    # Format C: [col_idx, row_idx, rho]          (3 kolom integer)

    sample = raw_data[:5]
    has_4col = all(len(r) >= 4 for r in sample)

    blocks = []
    for row in raw_data:
        try:
            if has_4col and len(row) >= 4:
                # block_id, x, depth, rho
                x = float(row[1])
                depth = float(row[2])
                rho = float(row[3])
            elif len(row) == 3:
                x = float(row[0])
                depth = float(row[1])
                rho = float(row[2])
            else:
                continue
            if rho > 0:
                blocks.append({"x": x, "depth": depth, "rho": rho})
        except:
            continue

    if not blocks:
        return

    # Dapatkan posisi unik
    xs = sorted(set(round(b["x"], 3) for b in blocks))
    depths = sorted(set(round(b["depth"], 3) for b in blocks))

    result["x_positions"] = xs
    result["depths"] = depths
    result["n_cols"] = len(xs)
    result["n_rows"] = len(depths)

    # Bangun matriks
    mat = np.zeros((len(depths), len(xs)))
    xi_map = {x: i for i, x in enumerate(xs)}
    di_map = {d: i for i, d in enumerate(depths)}

    for b in blocks:
        xi = xi_map.get(round(b["x"], 3))
        di = di_map.get(round(b["depth"], 3))
        if xi is not None and di is not None:
            mat[di][xi] = b["rho"]

    result["rho_matrix"] = mat
    result["blocks"] = blocks

    # Hitung rata-rata per lapisan
    layer_avgs = []
    for di, depth in enumerate(depths):
        row_vals = mat[di][mat[di] > 0]
        avg = float(np.mean(row_vals)) if len(row_vals) > 0 else 0.0
        layer_avgs.append({"depth": depth, "avg_rho": round(avg, 1)})
    result["layer_averages"] = layer_avgs


def _parse_simple_format(lines: list, result: dict) -> dict:
    """
    Fallback parser untuk format sederhana tanpa header iterasi yang jelas.
    """
    all_data = []
    for line in lines:
        parts = line.strip().split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except:
                pass
        if len(nums) >= 3 and all(n > 0 for n in nums[-3:]):
            all_data.append(nums[-3:])  # ambil 3 kolom terakhir: x, depth, rho

    if all_data:
        result["rms_history"] = [5.0]  # default jika tidak terdeteksi
        result["final_rms"] = 5.0
        result["iteration_used"] = 1
        _extract_grid(
            [{"x": r[0], "depth": r[1], "rho": r[2]} for r in all_data
             if len(r) >= 3],
            result
        )
        result["success"] = len(result.get("blocks", [])) > 0
        if not result["success"]:
            result["error"] = "Format file tidak dikenali. Pastikan file .INV dari RES2DINV."
    else:
        result["error"] = "Tidak ada data numerik yang valid ditemukan dalam file."

    return result


def generate_demo_inv() -> str:
    """
    Generate contoh file .INV sintetik untuk demo/testing.
    Mensimulasikan lintasan dengan anomali batuan keras di zona kanan.
    """
    n_cols = 15
    n_rows = 5
    electrode_spacing = 2.0

    # True resistivity model sintetik
    rho_true = np.array([
        [62,  75,  88,  95, 110, 145, 168, 210, 280, 420, 680, 850, 1100, 1350, 1480],
        [55,  68,  82, 100, 130, 180, 220, 310, 480, 620, 880, 1050,1300, 1450, 1550],
        [48,  60,  75,  95, 120, 160, 200, 280, 420, 580, 820,  980,1200, 1380, 1500],
        [45,  55,  70,  88, 110, 145, 185, 250, 380, 520, 760,  900,1100, 1280, 1420],
        [42,  50,  65,  80, 100, 130, 165, 220, 340, 460, 700,  840,1000, 1180, 1320],
    ], dtype=float)

    lines = []
    lines.append("Lintasan_Demo_Geolistrik_WS")
    lines.append(f"Unit electrode spacing = {electrode_spacing} m")
    lines.append("Array type: Wenner-Schlumberger (7)")
    lines.append(f"Number of electrodes: {n_cols + 1}")
    lines.append("")

    rms_vals = [21.4, 10.8, 6.2, 4.1, 3.2]
    depths = [0.6, 1.8, 3.0, 4.2, 5.4]

    for it, rms in enumerate(rms_vals, 1):
        lines.append(f"Iteration {it}   RMS error = {rms}%")
        blk = 1
        for ri, depth in enumerate(depths):
            for ci in range(n_cols):
                x_pos = ci * electrode_spacing + electrode_spacing
                noise = np.random.uniform(0.95, 1.05) if it < len(rms_vals) else 1.0
                rho = rho_true[ri][ci] * noise
                lines.append(f"  {blk:4d}  {x_pos:8.2f}  {depth:8.3f}  {rho:10.2f}")
                blk += 1
        lines.append("")

    return "\n".join(lines)


def get_array_name(code: int) -> str:
    mapping = {1: "Wenner", 3: "Dipole-Dipole", 7: "Wenner-Schlumberger"}
    return mapping.get(code, "Wenner-Schlumberger")
