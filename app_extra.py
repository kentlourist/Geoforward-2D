"""
Modul tambahan: export laporan HTML dan utilitas UI
"""
import numpy as np
import pandas as pd
from forward_model import classify_material


def build_html_report(inv_data: dict, fwd_results: dict) -> str:
    """Generate laporan HTML lengkap dari hasil analisis."""
    rho_mat   = inv_data.get("rho_matrix")
    depths    = inv_data.get("depths", [])
    filename  = inv_data.get("filename", "unknown")
    final_rms = inv_data.get("final_rms", 0)
    rms_hist  = inv_data.get("rms_history", [])
    n_elec    = inv_data.get("n_electrodes", 0)
    spacing   = inv_data.get("electrode_spacing", 0)

    # Layer averages
    layer_rows = ""
    if rho_mat is not None and len(depths) > 0:
        for ri, d in enumerate(depths):
            if ri < rho_mat.shape[0]:
                row = rho_mat[ri][rho_mat[ri] > 0]
                avg = float(np.mean(row)) if len(row) > 0 else 0
                mat, color = classify_material(avg)
                layer_rows += f"""
                <tr>
                  <td>Lapisan {ri+1}</td>
                  <td>{d:.1f} m</td>
                  <td style="font-weight:600;color:{color}">{avg:.1f}</td>
                  <td><span style="background:{color}22;color:{color};
                      padding:2px 8px;border-radius:12px;font-size:12px">{mat}</span></td>
                </tr>"""

    # Forward summary rows
    fwd_rows = ""
    for key, res in fwd_results.items():
        dp = res.datum_points
        if dp is None or len(dp) == 0:
            continue
        s = res.electrode_spacing
        arr = res.array_name
        fwd_rows += f"""
        <tr>
          <td>{arr}</td>
          <td>{s} m</td>
          <td>{len(dp)}</td>
          <td>{dp[:,2].min():.1f}</td>
          <td>{dp[:,2].max():.1f}</td>
          <td>{dp[:,2].mean():.1f}</td>
          <td>{dp[:,1].max():.2f} m</td>
        </tr>"""

    # RMS history table
    rms_rows = ""
    for i, v in enumerate(rms_hist, 1):
        status = "✅ Konvergen" if v <= 5 else "🔄 Proses"
        color  = "#0F6E56" if v <= 5 else "#185FA5"
        rms_rows += f"""
        <tr>
          <td>Iterasi {i}</td>
          <td style="font-weight:600;color:{color}">{v:.1f}%</td>
          <td style="color:{color}">{status}</td>
        </tr>"""

    rms_status_txt = "Konvergen (RMS < 5%)" if final_rms and final_rms <= 5 else "Perlu evaluasi"
    rms_color      = "#0F6E56" if final_rms and final_rms <= 5 else "#D85A30"

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laporan GeoForward 2D — {filename}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;color:#1a1a1a;background:#f8fafc;padding:32px}}
  .page{{max-width:900px;margin:0 auto;background:white;border-radius:16px;
         box-shadow:0 2px 20px rgba(0,0,0,.08);overflow:hidden}}
  .hdr{{background:linear-gradient(135deg,#0a2744 0%,#185FA5 60%,#0F6E56 100%);
        color:white;padding:32px 36px}}
  .hdr h1{{font-size:1.8rem;font-weight:700;margin-bottom:6px}}
  .hdr p{{opacity:.82;font-size:.95rem}}
  .body{{padding:32px 36px}}
  .sec{{margin-bottom:32px}}
  .sec-title{{font-size:1rem;font-weight:600;color:#0a2744;border-left:4px solid #185FA5;
              padding-left:12px;margin-bottom:16px}}
  .kv-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}}
  .kv{{background:#f0f7ff;border:1px solid #B5D4F4;border-radius:10px;padding:12px 16px}}
  .kv .v{{font-size:1.3rem;font-weight:600;color:#185FA5}}
  .kv .k{{font-size:.7rem;color:#6b7280;margin-top:3px;text-transform:uppercase}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th{{background:#f0f7ff;color:#0a2744;padding:9px 12px;text-align:left;
      border-bottom:2px solid #B5D4F4;font-weight:600}}
  td{{padding:8px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle}}
  tr:hover td{{background:#fafafa}}
  .badge-ok{{background:#E1F5EE;color:#085041;padding:2px 10px;
             border-radius:20px;font-size:.78rem;font-weight:500}}
  .footer{{text-align:center;padding:20px;color:#9ca3af;font-size:.75rem;
           border-top:1px solid #f0f0f0;margin-top:24px}}
  @media print{{body{{background:white;padding:0}}
    .page{{box-shadow:none;border-radius:0}}.no-print{{display:none}}}}
</style>
</head>
<body>
<div class="page">
  <div class="hdr">
    <h1>🌍 Laporan GeoForward 2D</h1>
    <p>Forward Modelling Geolistrik Tahanan Jenis 2D &nbsp;·&nbsp; {filename}</p>
  </div>
  <div class="body">

    <!-- Metadata -->
    <div class="sec">
      <div class="sec-title">Informasi File Inversi</div>
      <div class="kv-grid">
        <div class="kv"><div class="v">{filename}</div><div class="k">Nama file</div></div>
        <div class="kv"><div class="v">{n_elec}</div><div class="k">Jumlah elektroda</div></div>
        <div class="kv"><div class="v">{spacing} m</div><div class="k">Spasi elektroda</div></div>
        <div class="kv"><div class="v" style="color:{rms_color}">{final_rms:.1f}%</div>
             <div class="k">RMS Error akhir</div></div>
        <div class="kv"><div class="v">{inv_data.get('iteration_used','-')}</div>
             <div class="k">Iterasi digunakan</div></div>
        <div class="kv"><div class="v" style="color:{rms_color}">{rms_status_txt}</div>
             <div class="k">Status konvergensi</div></div>
      </div>
    </div>

    <!-- RMS History -->
    <div class="sec">
      <div class="sec-title">Riwayat Konvergensi Inversi</div>
      <table><thead><tr><th>Iterasi</th><th>RMS Error</th><th>Status</th></tr></thead>
      <tbody>{rms_rows}</tbody></table>
    </div>

    <!-- Layer model -->
    <div class="sec">
      <div class="sec-title">Model Resistivitas Per Lapisan (dari RES2DINV)</div>
      <table>
        <thead><tr><th>Lapisan</th><th>Kedalaman</th>
          <th>ρ rata-rata (Ω·m)</th><th>Interpretasi Litologi</th></tr></thead>
        <tbody>{layer_rows}</tbody>
      </table>
    </div>

    <!-- Forward results -->
    <div class="sec">
      <div class="sec-title">Ringkasan Hasil Forward Modelling</div>
      <table>
        <thead><tr><th>Konfigurasi</th><th>Spasi</th><th>Datum</th>
          <th>ρa min (Ω·m)</th><th>ρa maks (Ω·m)</th>
          <th>ρa rata (Ω·m)</th><th>Depth maks</th></tr></thead>
        <tbody>{fwd_rows if fwd_rows else "<tr><td colspan='7' style='text-align:center;color:#9ca3af'>Belum ada hasil forward modelling</td></tr>"}</tbody>
      </table>
    </div>

    <!-- Rekomendasi -->
    <div class="sec">
      <div class="sec-title">Panduan Interpretasi Hasil</div>
      <table>
        <thead><tr><th>Konfigurasi</th><th>Sensitivitas</th>
          <th>Rasio S/N</th><th>Rekomendasi penggunaan</th></tr></thead>
        <tbody>
          <tr><td>Wenner-Schlumberger</td><td>Vertikal + lateral</td>
              <td>Sedang</td><td>Serbaguna — paling umum untuk survei 2D</td></tr>
          <tr><td>Wenner</td><td>Vertikal</td>
              <td>Tinggi</td><td>Lapisan horizontal, area dengan noise tinggi</td></tr>
          <tr><td>Dipole-Dipole</td><td>Lateral</td>
              <td>Rendah</td><td>Deteksi patahan dan kontak litologi vertikal</td></tr>
        </tbody>
      </table>
    </div>

    <div class="footer">
      Dibuat dengan GeoForward 2D &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp;
      Referensi: Loke (2004) RES2DMOD Manual
    </div>
  </div>
</div>
</body>
</html>"""
    return html
