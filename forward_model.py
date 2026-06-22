"""
Modul Forward Modelling Geolistrik 2D
Menghitung apparent resistivity (ρa) dari model true resistivity
untuk berbagai konfigurasi elektroda dan spasi.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ─── Konstanta konfigurasi ────────────────────────────────────────────────────

ARRAY_CONFIGS = {
    "Wenner-Schlumberger": {
        "code": 7,
        "k_formula": "π·n(n+1)·a",
        "description": "Sensitif vertikal + lateral, serbaguna",
        "max_n_default": 6,
        "depth_factor": 0.519,
    },
    "Wenner": {
        "code": 1,
        "k_formula": "2πa",
        "description": "Sensitif vertikal, sinyal kuat",
        "max_n_default": 1,
        "depth_factor": 0.500,
    },
    "Dipole-Dipole": {
        "code": 3,
        "k_formula": "πn(n+1)(n+2)·a",
        "description": "Sensitif lateral, deteksi struktur vertikal",
        "max_n_default": 6,
        "depth_factor": 0.300,
    },
}


@dataclass
class ForwardResult:
    """Hasil forward modelling satu konfigurasi."""
    array_name: str
    electrode_spacing: float
    n_max: int
    datum_points: np.ndarray    # shape (N, 3): [x_mid, pseudo_depth, rho_a]
    n_electrodes: int
    geometric_factors: np.ndarray  # faktor K untuk setiap datum


# ─── Fungsi geometric factor ─────────────────────────────────────────────────

def K_wenner(a: float, n: float = 1) -> float:
    """K untuk Wenner: 2πa"""
    return 2 * np.pi * a


def K_wenner_schlumberger(a: float, n: float) -> float:
    """K untuk Wenner-Schlumberger: πn(n+1)a"""
    return np.pi * n * (n + 1) * a


def K_dipole_dipole(a: float, n: float) -> float:
    """K untuk Dipole-Dipole: πn(n+1)(n+2)a"""
    return np.pi * n * (n + 1) * (n + 2) * a


K_FUNCTIONS = {
    "Wenner": K_wenner,
    "Wenner-Schlumberger": K_wenner_schlumberger,
    "Dipole-Dipole": K_dipole_dipole,
}


# ─── Fungsi sensitivitas ─────────────────────────────────────────────────────

def compute_sensitivity_ws(x_c1, x_p1, x_p2, x_c2, x_blk, z_blk):
    """
    Sensitivitas Wenner-Schlumberger untuk satu blok model.
    Pendekatan semi-analitik berdasarkan kernel sensitivitas Geyer (1983).
    """
    def contrib(x_src, x_recv, x_b, z_b):
        dx_s = x_b - x_src
        dx_r = x_b - x_recv
        r_s = np.sqrt(dx_s**2 + z_b**2) + 1e-10
        r_r = np.sqrt(dx_r**2 + z_b**2) + 1e-10
        return (1/r_s - 1/r_r) / (2 * np.pi)

    s = (contrib(x_c1, x_c2, x_blk, z_blk) *
         contrib(x_p1, x_p2, x_blk, z_blk))
    return abs(s)


def _build_pseudo_depth(array_name: str, a: float, n: float) -> float:
    """Kedalaman titik datum dalam pseudosection."""
    cfg = ARRAY_CONFIGS.get(array_name, ARRAY_CONFIGS["Wenner-Schlumberger"])
    return cfg["depth_factor"] * n * a


def _mid_point_ws(x_c1: float, a: float, n: float) -> float:
    """Titik tengah horizontal datum W-S."""
    return x_c1 + n * a + a / 2


def _mid_point_dd(x_c1: float, a: float, n: float) -> float:
    """Titik tengah horizontal datum Dipole-Dipole."""
    return x_c1 + a + n * a / 2 + a / 2


# ─── Fungsi utama forward modelling ──────────────────────────────────────────

def run_forward(
    rho_matrix: np.ndarray,
    x_positions: list,
    depths: list,
    array_name: str = "Wenner-Schlumberger",
    electrode_spacing: float = 2.0,
    n_max: int = 6,
    n_electrodes: int = 16,
    noise_level: float = 0.0,
) -> ForwardResult:
    """
    Jalankan forward modelling 2D.

    Parameters
    ----------
    rho_matrix    : array [n_rows × n_cols] nilai true resistivity (Ω·m)
    x_positions   : posisi x pusat blok (m)
    depths        : kedalaman pusat blok (m)
    array_name    : "Wenner-Schlumberger" | "Wenner" | "Dipole-Dipole"
    electrode_spacing : spasi elektroda a (m)
    n_max         : level n maksimum
    n_electrodes  : jumlah elektroda
    noise_level   : level noise relatif (0 = tanpa noise, 0.05 = 5%)

    Returns
    -------
    ForwardResult berisi semua titik datum pseudosection
    """
    a = electrode_spacing
    cfg = ARRAY_CONFIGS.get(array_name, ARRAY_CONFIGS["Wenner-Schlumberger"])
    K_func = K_FUNCTIONS.get(array_name, K_wenner_schlumberger)

    # Posisi elektroda di permukaan
    elec_x = np.array([i * a for i in range(n_electrodes)])
    n_elec = len(elec_x)

    datum_list = []
    K_list = []

    # ── Wenner-Schlumberger ──
    if array_name == "Wenner-Schlumberger":
        for n in range(1, n_max + 1):
            K = K_func(a, n)
            for i in range(n_elec - (2 * n + 1)):
                x_c1 = elec_x[i]
                x_p1 = elec_x[i + n]
                x_p2 = elec_x[i + n + 1]
                x_c2 = elec_x[i + 2 * n + 1]

                x_mid = (x_p1 + x_p2) / 2
                z_mid = _build_pseudo_depth(array_name, a, n)

                rho_a = _weighted_rho(
                    rho_matrix, x_positions, depths,
                    x_c1, x_p1, x_p2, x_c2, array_name
                )

                if noise_level > 0:
                    rho_a *= (1 + np.random.uniform(-noise_level, noise_level))

                datum_list.append([x_mid, z_mid, rho_a])
                K_list.append(K)

    # ── Wenner ──
    elif array_name == "Wenner":
        K = K_func(a)
        for i in range(n_elec - 3):
            x_c1 = elec_x[i]
            x_p1 = elec_x[i + 1]
            x_p2 = elec_x[i + 2]
            x_c2 = elec_x[i + 3]

            x_mid = (x_c1 + x_c2) / 2
            z_mid = _build_pseudo_depth(array_name, a, 1)

            rho_a = _weighted_rho(
                rho_matrix, x_positions, depths,
                x_c1, x_p1, x_p2, x_c2, array_name
            )
            if noise_level > 0:
                rho_a *= (1 + np.random.uniform(-noise_level, noise_level))

            datum_list.append([x_mid, z_mid, rho_a])
            K_list.append(K)

    # ── Dipole-Dipole ──
    elif array_name == "Dipole-Dipole":
        for n in range(1, n_max + 1):
            K = K_func(a, n)
            for i in range(n_elec - n - 2):
                x_c1 = elec_x[i]
                x_c2 = elec_x[i + 1]
                x_p1 = elec_x[i + n + 1]
                x_p2 = elec_x[i + n + 2]

                x_mid = (x_c2 + x_p1) / 2
                z_mid = _build_pseudo_depth(array_name, a, n)

                rho_a = _weighted_rho(
                    rho_matrix, x_positions, depths,
                    x_c1, x_p1, x_p2, x_c2, array_name
                )
                if noise_level > 0:
                    rho_a *= (1 + np.random.uniform(-noise_level, noise_level))

                datum_list.append([x_mid, z_mid, rho_a])
                K_list.append(K)

    datum_arr = np.array(datum_list) if datum_list else np.zeros((1, 3))
    K_arr = np.array(K_list) if K_list else np.array([1.0])

    return ForwardResult(
        array_name=array_name,
        electrode_spacing=a,
        n_max=n_max,
        datum_points=datum_arr,
        n_electrodes=n_electrodes,
        geometric_factors=K_arr,
    )


def _weighted_rho(rho_matrix, x_positions, depths,
                  x_c1, x_p1, x_p2, x_c2, array_name) -> float:
    """
    Hitung apparent resistivity berbobot dari model blok.
    Menggunakan interpolasi berbobot jarak terbalik (IDW).
    """
    x_mid = (x_p1 + x_p2) / 2
    spread = abs(x_c2 - x_c1)

    if rho_matrix is None or rho_matrix.size == 0:
        return 100.0

    x_arr = np.array(x_positions)
    z_arr = np.array(depths)

    # Rentang investigasi
    x_min = min(x_c1, x_p1, x_p2, x_c2)
    x_max = max(x_c1, x_p1, x_p2, x_c2)

    weighted_sum = 0.0
    weight_total = 0.0

    n_rows, n_cols = rho_matrix.shape

    for ri in range(n_rows):
        for ci in range(n_cols):
            if ci >= len(x_arr) or ri >= len(z_arr):
                continue
            xb = x_arr[ci]
            zb = z_arr[ri]
            rho = rho_matrix[ri, ci]
            if rho <= 0:
                continue

            # Bobot berdasarkan proximity ke titik pengukuran
            dx = xb - x_mid
            dz = zb
            dist = np.sqrt(dx**2 + dz**2) + 1e-6

            # Sensitivitas lebih tinggi untuk blok dekat tengah dan dangkal
            if x_min <= xb <= x_max:
                w = 1.0 / (dist ** 1.5)
            else:
                # Atenuasi untuk blok di luar rentang elektroda
                excess = min(abs(xb - x_min), abs(xb - x_max))
                w = 1.0 / (dist ** 1.5) * np.exp(-excess / (spread + 1e-6))

            weighted_sum += rho * w
            weight_total += w

    if weight_total > 0:
        return weighted_sum / weight_total
    return float(np.mean(rho_matrix[rho_matrix > 0])) if np.any(rho_matrix > 0) else 100.0


def compute_layer_averages(rho_matrix: np.ndarray, depths: list) -> list:
    """Hitung rata-rata ρ per lapisan."""
    result = []
    for ri, d in enumerate(depths):
        if ri < rho_matrix.shape[0]:
            row = rho_matrix[ri][rho_matrix[ri] > 0]
            avg = float(np.mean(row)) if len(row) > 0 else 0.0
            result.append({
                "layer": ri + 1,
                "depth": d,
                "avg_rho": round(avg, 1),
                "min_rho": round(float(np.min(row)), 1) if len(row) > 0 else 0,
                "max_rho": round(float(np.max(row)), 1) if len(row) > 0 else 0,
            })
    return result


def classify_material(rho: float) -> Tuple[str, str]:
    """Klasifikasi material berdasarkan nilai resistivitas."""
    if rho < 10:
        return "Air / Lempung jenuh", "#1565C0"
    elif rho < 50:
        return "Lempung basah", "#1976D2"
    elif rho < 150:
        return "Lempung / Pasir lembab", "#1D9E75"
    elif rho < 400:
        return "Pasir / Kerikil", "#2E7D32"
    elif rho < 800:
        return "Kerikil / Pasir kering", "#EF9F27"
    elif rho < 2000:
        return "Batuan lapuk / Batupasir", "#D85A30"
    else:
        return "Batuan keras / Granit", "#B71C1C"
