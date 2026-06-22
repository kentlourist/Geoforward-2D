"""
GeoForward 2D — Aplikasi Web Forward Modelling Geolistrik
Berbasis Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import io, time

from inv_parser     import parse_inv_file, generate_demo_inv
from forward_model  import run_forward, compute_layer_averages, classify_material, ARRAY_CONFIGS
from plotting       import (plot_pseudosection, plot_true_section, plot_rms_convergence,
                            plot_comparison_spasi, plot_comparison_array, plot_layer_bar)
from app_extra      import build_html_report

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoForward 2D",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
/* Header */
.geo-hdr{background:linear-gradient(135deg,#0a2744 0%,#185FA5 55%,#0F6E56 100%);
  border-radius:14px;padding:26px 30px;margin-bottom:20px;color:white}
.geo-hdr h1{font-size:2rem;font-weight:700;margin:0 0 5px;letter-spacing:-.5px}
.geo-hdr p{font-size:.95rem;margin:0;opacity:.82}
.geo-badge{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);
  border-radius:20px;padding:3px 12px;font-size:.72rem;font-weight:500;margin:8px 4px 0 0}
/* Section headers */
.sec-head{font-size:1rem;font-weight:600;color:#0a2744;border-left:4px solid #185FA5;
  padding-left:12px;margin:22px 0 10px}
.sec-sub{font-size:.82rem;color:#6b7280;margin:-6px 0 12px 16px}
/* Cards */
.info-box{background:#EFF6FF;border:1px solid #B5D4F4;border-radius:10px;
  padding:13px 16px;margin:10px 0;font-size:.86rem;color:#0C447C;line-height:1.65}
.warn-box{background:#FAEEDA;border:1px solid #FAC775;border-radius:10px;
  padding:11px 16px;margin:8px 0;font-size:.84rem;color:#633806}
.ok-box  {background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;
  padding:11px 16px;margin:8px 0;font-size:.84rem;color:#085041}
/* Layer rows */
.layer-row{display:flex;align-items:center;gap:10px;padding:8px 12px;
  border-radius:8px;margin-bottom:5px;border:1px solid #e8eaf0;background:white}
.layer-dot{width:17px;height:17px;border-radius:50%;flex-shrink:0}
.layer-info{flex:1}
.layer-name{font-size:.82rem;font-weight:500;color:#1a1a1a}
.layer-val{font-size:.73rem;color:#6b7280}
.layer-rho{font-size:.88rem;font-weight:600;color:#185FA5;min-width:74px;
  text-align:right;font-family:'JetBrains Mono',monospace}
/* Stat card */
.stat-row{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.stat-card{background:white;border:1px solid #e8eaf0;border-radius:10px;
  padding:12px 16px;flex:1;min-width:110px}
.stat-card .sv{font-size:1.4rem;font-weight:600;color:#185FA5;line-height:1.2}
.stat-card .sl{font-size:.68rem;color:#6b7280;margin-top:3px;text-transform:uppercase;letter-spacing:.05em}
/* Rec cards */
.rec-card{border-radius:10px;padding:14px;height:100%}
/* Step badges */
.step-pill{display:inline-flex;align-items:center;gap:6px;background:#0a2744;
  color:white;border-radius:20px;padding:4px 14px;font-size:.8rem;font-weight:500;margin-bottom:8px}
/* Guide steps */
.guide-step{display:flex;gap:14px;margin-bottom:16px;align-items:flex-start}
.guide-num{background:#185FA5;color:white;border-radius:50%;width:28px;height:28px;
  display:flex;align-items:center;justify-content:center;font-size:.82rem;font-weight:600;flex-shrink:0;margin-top:2px}
.guide-body{flex:1}
.guide-title{font-size:.9rem;font-weight:600;color:#0a2744;margin-bottom:3px}
.guide-desc{font-size:.82rem;color:#374151;line-height:1.55}
.guide-code{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;
  padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:.78rem;
  color:#1e3a5f;margin-top:6px;display:block}
/* Footer */
.footer{text-align:center;color:#9ca3af;font-size:.75rem;padding:18px 0 6px;
  border-top:1px solid #f0f0f0;margin-top:28px}
/* Hide default streamlit header margin */
div[data-testid="stSidebarNav"]{display:none}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "inv_parsed": None,
    "forward_results": {},
    "demo_loaded": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="geo-hdr">
  <h1>🌍 GeoForward 2D</h1>
  <p>Forward modelling geolistrik tahanan jenis 2D — simulasi otomatis variasi spasi dan konfigurasi elektroda</p>
  <span class="geo-badge">Upload .INV</span>
  <span class="geo-badge">Wenner-Schlumberger</span>
  <span class="geo-badge">Wenner</span>
  <span class="geo-badge">Dipole-Dipole</span>
  <span class="geo-badge">Multi-spasi</span>
  <span class="geo-badge">Export Laporan</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")
    st.markdown("---")

    # ── Upload ──
    st.markdown("**📂 1. Sumber Data**")
    uploaded = st.file_uploader(
        "Upload file .INV (RES2DINV)",
        type=["inv","INV","txt"],
        help="Output file .INV dari RES2DINV versi 3.x – 10.x",
    )

    col_d1, col_d2 = st.columns([1,1])
    with col_d1:
        demo_btn = st.button("🧪 Demo", use_container_width=True,
                             help="Muat data sintetik lintasan 30m")
    with col_d2:
        clear_btn = st.button("🗑 Reset", use_container_width=True,
                              help="Hapus semua data dan mulai ulang")

    if clear_btn:
        st.session_state.inv_parsed      = None
        st.session_state.forward_results = {}
        st.session_state.demo_loaded     = False
        st.rerun()

    if demo_btn:
        with st.spinner("Memuat data demo..."):
            content = generate_demo_inv()
            parsed  = parse_inv_file(content)
            parsed["filename"] = "demo_lintasan_30m_WS.INV"
            parsed["is_demo"]  = True
            st.session_state.inv_parsed      = parsed
            st.session_state.demo_loaded     = True
            st.session_state.forward_results = {}
        st.success("✅ Data demo dimuat!")

    if uploaded and not st.session_state.demo_loaded:
        content = uploaded.read().decode("utf-8", errors="replace")
        with st.spinner("Memproses file..."):
            parsed = parse_inv_file(content)
            parsed["filename"] = uploaded.name
            parsed["is_demo"]  = False
            st.session_state.inv_parsed      = parsed
            st.session_state.forward_results = {}
        if parsed["success"]:
            st.success("✅ File berhasil dibaca!")
        else:
            st.error(f"❌ {parsed.get('error','Error tidak diketahui')}")

    # Status file
    inv = st.session_state.inv_parsed
    if inv and inv.get("success"):
        rms = inv.get("final_rms", 0)
        rms_ok = rms and rms <= 5
        st.markdown(f"""
        <div style='background:{"#E1F5EE" if rms_ok else "#FAEEDA"};
          border:1px solid {"#9FE1CB" if rms_ok else "#FAC775"};
          border-radius:8px;padding:8px 12px;font-size:.8rem;
          color:{"#085041" if rms_ok else "#633806"}'>
          {'✅' if rms_ok else '⚠️'} <b>{inv.get("filename","")}</b><br>
          RMS: {rms:.1f}% &nbsp;·&nbsp; {inv.get("n_cols",0)}×{inv.get("n_rows",0)} blok &nbsp;·&nbsp; a={inv.get("electrode_spacing",0)}m
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔧 2. Parameter Forward**")

    spasi_opts = st.multiselect(
        "Spasi elektroda (m)",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3],
        help="Semua spasi yang dipilih akan disimulasikan",
    )
    array_opts = st.multiselect(
        "Konfigurasi elektroda",
        options=["Wenner-Schlumberger","Wenner","Dipole-Dipole"],
        default=["Wenner-Schlumberger","Wenner","Dipole-Dipole"],
    )

    st.markdown("**🎛 3. Opsi Lanjutan**")
    noise_pct  = st.slider("Noise sintetik (%)", 0, 20, 0, 1,
                           help="Tambahkan noise acak ke data forward")
    show_vals  = st.checkbox("Tampilkan nilai ρa di plot", False)
    col_lo, col_hi = st.columns(2)
    with col_lo:
        rho_min_inp = st.number_input("ρ min (Ω·m)", 1.0, 9999.0, 10.0, 1.0)
    with col_hi:
        rho_max_inp = st.number_input("ρ maks (Ω·m)", 1.0, 99999.0, 2000.0, 100.0)

    st.markdown("---")
    can_run = (inv is not None and inv.get("success") and
               len(spasi_opts) > 0 and len(array_opts) > 0)
    run_btn = st.button("▶ Jalankan Forward Modelling",
                        use_container_width=True, type="primary",
                        disabled=not can_run)

    if not can_run and inv is None:
        st.markdown('<div class="warn-box">Upload file .INV atau gunakan data demo terlebih dahulu.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.73rem;color:#9ca3af;line-height:1.65'>
    <b>GeoForward 2D</b><br>
    Kalkulasi: Weighted IDW<br>
    Ref: Loke (2004) RES2DMOD Manual<br>
    Python · Streamlit · Plotly
    </div>""", unsafe_allow_html=True)

# ── Jalankan forward ──────────────────────────────────────────────────────────
if run_btn and inv:
    rho_mat = inv.get("rho_matrix")
    x_pos   = inv.get("x_positions", [])
    depths  = inv.get("depths", [])
    n_elec  = inv.get("n_electrodes", 16) or 16

    if rho_mat is None or rho_mat.size == 0:
        st.error("Model resistivitas kosong. Periksa format file .INV.")
    else:
        total = len(spasi_opts) * len(array_opts)
        prog  = st.progress(0, "Menjalankan simulasi forward modelling...")
        done  = 0
        results = {}
        t0 = time.time()

        for s in spasi_opts:
            for arr in array_opts:
                key = f"{s}_{arr}"
                cfg = ARRAY_CONFIGS.get(arr, {})
                res = run_forward(
                    rho_matrix=rho_mat, x_positions=x_pos, depths=depths,
                    array_name=arr, electrode_spacing=float(s),
                    n_max=cfg.get("max_n_default", 6),
                    n_electrodes=n_elec,
                    noise_level=noise_pct / 100,
                )
                results[key] = res
                done += 1
                prog.progress(done / total,
                              f"Simulasi {done}/{total}: {arr} a={s}m ✓")

        st.session_state.forward_results = results
        prog.empty()
        elapsed = time.time() - t0
        st.success(f"✅ {total} simulasi selesai dalam {elapsed:.1f} detik! Lihat hasil di tab di bawah.")

# ── Landing page jika belum ada data ─────────────────────────────────────────
inv = st.session_state.inv_parsed
if inv is None:
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📂","Upload File .INV","File output RES2DINV langsung di-parse otomatis. Semua format v3.x–10.x didukung."),
        ("⚙️","Atur Parameter","Pilih variasi spasi (1–5 m) dan konfigurasi elektroda. Bisa pilih lebih dari satu."),
        ("📊","Analisis Hasil","Pseudosection interaktif, komparasi visual, tabel statistik, dan export laporan HTML."),
    ]
    for col, (ic, title, desc) in zip([c1,c2,c3], cards):
        with col:
            st.markdown(f"""
            <div style='background:white;border:1px solid #e8eaf0;border-radius:12px;
              padding:22px;text-align:center;height:180px'>
              <div style='font-size:2rem'>{ic}</div>
              <b style='font-size:.95rem;color:#0a2744'>{title}</b>
              <p style='font-size:.82rem;color:#6b7280;margin-top:8px'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">💡 <b>Belum punya file .INV?</b> Klik tombol <b>🧪 Demo</b> di sidebar untuk mencoba aplikasi dengan data sintetik lintasan 30 m.</div>',
                unsafe_allow_html=True)

    # Tampilkan tab panduan saja
    with st.expander("📖 Panduan Lengkap Penggunaan Aplikasi", expanded=True):
        steps = [
            ("Upload / Demo","Klik <b>🧪 Demo</b> di sidebar untuk data contoh, atau upload file <code>.INV</code> hasil RES2DINV Anda.",
             "File .INV → sidebar → Upload file .INV dari RES2DINV"),
            ("Pilih Parameter","Di sidebar, centang variasi <b>Spasi elektroda</b> dan <b>Konfigurasi</b> yang ingin dibandingkan.",
             "Spasi: [1, 2, 3] m  |  Konfigurasi: [WS, Wenner, DD]"),
            ("Jalankan Simulasi","Klik tombol <b>▶ Jalankan Forward Modelling</b>. Semua kombinasi dijalankan otomatis.",
             "Total simulasi = jumlah spasi × jumlah konfigurasi"),
            ("Analisis Hasil","Gunakan tab di bawah untuk melihat penampang, membandingkan variasi, dan export laporan.",
             "Tab: Data Inversi → Model → Forward → Komparasi Spasi → Komparasi Konfigurasi → Laporan"),
        ]
        for i, (title, desc, code) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="guide-step">
              <div class="guide-num">{i}</div>
              <div class="guide-body">
                <div class="guide-title">{title}</div>
                <div class="guide-desc">{desc}</div>
                <code class="guide-code">{code}</code>
              </div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Tab utama ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data Inversi",
    "🗺️ Model True ρ",
    "🔄 Forward Modelling",
    "📊 Komparasi Spasi",
    "📐 Komparasi Konfigurasi",
    "📄 Laporan & Export",
])

fwd = st.session_state.forward_results

# ═══════════════════════════════════════════════════
# TAB 1 — DATA INVERSI
# ═══════════════════════════════════════════════════
with tab1:
    if inv.get("is_demo"):
        st.markdown('<div class="warn-box">📌 Mode demo aktif — data sintetik lintasan 30 m, Wenner-Schlumberger.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Ringkasan File Inversi</div>', unsafe_allow_html=True)

    rms   = inv.get("final_rms", 0)
    rms_ok = rms and rms <= 5
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("File", inv.get("filename","-")[:20])
    c2.metric("Iterasi",  inv.get("iteration_used","-"))
    c3.metric("RMS Error", f"{rms:.1f}%" if rms else "-",
              delta="Konvergen ✓" if rms_ok else "Perlu evaluasi",
              delta_color="normal" if rms_ok else "inverse")
    c4.metric("Grid blok", f"{inv.get('n_cols',0)} × {inv.get('n_rows',0)}")
    c5.metric("Spasi a", f"{inv.get('electrode_spacing','-')} m")

    # RMS chart
    if inv.get("rms_history"):
        st.markdown('<div class="sec-head">Konvergensi Iterasi Inversi</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_rms_convergence(inv["rms_history"]), use_container_width=True)
        if rms_ok:
            st.markdown(f'<div class="ok-box">✅ RMS Error akhir <b>{rms:.1f}%</b> berada di bawah batas 5%. Model inversi valid dan siap digunakan sebagai acuan forward modelling.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">⚠️ RMS Error {rms:.1f}% masih di atas 5%. Pertimbangkan untuk menambah iterasi di RES2DINV atau memeriksa kualitas data lapangan.</div>',
                        unsafe_allow_html=True)

    # Layer averages
    rho_mat = inv.get("rho_matrix")
    depths  = inv.get("depths", [])
    if rho_mat is not None and len(depths) > 0:
        st.markdown('<div class="sec-head">Nilai ρ Per Lapisan</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Rata-rata resistivitas dari file .INV — akan digunakan sebagai model acuan forward modelling</div>',
                    unsafe_allow_html=True)
        la = compute_layer_averages(rho_mat, depths)
        col_l, col_r = st.columns([1, 2])
        with col_l:
            for l in la:
                mat, color = classify_material(l["avg_rho"])
                st.markdown(f"""
                <div class="layer-row">
                  <div class="layer-dot" style="background:{color}"></div>
                  <div class="layer-info">
                    <div class="layer-name">Lapisan {l['layer']} &nbsp;(z = {l['depth']:.1f} m)</div>
                    <div class="layer-val">{mat}</div>
                  </div>
                  <div class="layer-rho">{l['avg_rho']:.0f} Ω·m</div>
                </div>""", unsafe_allow_html=True)
        with col_r:
            st.plotly_chart(plot_layer_bar(la), use_container_width=True)

        # Tabel blok detail
        with st.expander("🔢 Tabel nilai ρ per blok lengkap"):
            x_pos = inv.get("x_positions", list(range(inv.get("n_cols",0))))
            rows  = []
            for ri, d in enumerate(depths):
                if ri < rho_mat.shape[0]:
                    row = {"Kedalaman (m)": d}
                    for ci, x in enumerate(x_pos):
                        if ci < rho_mat.shape[1]:
                            row[f"x={x:.1f}m"] = round(float(rho_mat[ri,ci]),1)
                    valid = rho_mat[ri][rho_mat[ri]>0]
                    row["Rata-rata (Ω·m)"] = round(float(np.mean(valid)),1) if len(valid)>0 else 0
                    rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=220)
            st.download_button("⬇ Download CSV",
                               data=df.to_csv(index=False),
                               file_name="rho_blok.csv", mime="text/csv")

# ═══════════════════════════════════════════════════
# TAB 2 — MODEL TRUE ρ
# ═══════════════════════════════════════════════════
with tab2:
    rho_mat = inv.get("rho_matrix")
    x_pos   = inv.get("x_positions", [])
    depths  = inv.get("depths", [])

    if rho_mat is None or rho_mat.size == 0:
        st.warning("Model resistivitas tidak tersedia.")
    else:
        st.markdown('<div class="sec-head">Penampang True Resistivity 2D — Hasil Inversi RES2DINV</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Distribusi resistivitas sesungguhnya yang menjadi model acuan seluruh simulasi forward</div>',
                    unsafe_allow_html=True)

        valid = rho_mat[rho_mat > 0].flatten()
        rlo = float(np.percentile(valid, 2)) if len(valid)>0 else 10.0
        rhi = float(np.percentile(valid, 98)) if len(valid)>0 else 2000.0
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            rng = st.slider("Rentang skala warna (Ω·m)", 1.0,
                            max(10000., float(rho_mat.max())),
                            (max(1., rlo), min(10000., rhi)), format="%.0f")
        with col_s2:
            show_interp = st.checkbox("Tampilkan anotasi interpretasi", True)

        st.plotly_chart(
            plot_true_section(rho_mat, x_pos, depths,
                              title="Penampang True Resistivity 2D",
                              rho_min=rng[0], rho_max=rng[1], height=400),
            use_container_width=True)

        # Statistik
        st.markdown('<div class="sec-head">Statistik & Interpretasi Geologi</div>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("ρ minimum",  f"{valid.min():.1f} Ω·m"  if len(valid)>0 else "-")
        c2.metric("ρ maksimum", f"{valid.max():.1f} Ω·m"  if len(valid)>0 else "-")
        c3.metric("ρ rata-rata",f"{valid.mean():.1f} Ω·m" if len(valid)>0 else "-")
        c4.metric("ρ median",   f"{np.median(valid):.1f} Ω·m" if len(valid)>0 else "-")
        c5.metric("Total blok", f"{rho_mat.size}")

        la = compute_layer_averages(rho_mat, depths)
        df_interp = pd.DataFrame([{
            "Lapisan": f"Lap.{l['layer']} (z={l['depth']:.1f}m)",
            "ρ rata-rata (Ω·m)": l["avg_rho"],
            "ρ min": l["min_rho"], "ρ maks": l["max_rho"],
            "Litologi": classify_material(l["avg_rho"])[0],
        } for l in la])
        st.dataframe(df_interp, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════
# TAB 3 — FORWARD MODELLING (individual)
# ═══════════════════════════════════════════════════
with tab3:
    if not fwd:
        st.markdown('<div class="info-box">💡 Atur parameter di sidebar lalu klik <b>▶ Jalankan Forward Modelling</b>.</div>',
                    unsafe_allow_html=True)
        # Penjelasan alur
        st.markdown('<div class="sec-head">Mekanisme Forward Modelling di Aplikasi Ini</div>',
                    unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        for col, (icon,title,body) in zip([col1,col2,col3],[
            ("📥","Input","Model true ρ dari .INV · Spasi elektroda · Konfigurasi · N-level maks"),
            ("⚙️","Kalkulasi","Posisi kombinasi C1-P1-P2-C2 · Faktor K · Interpolasi IDW · Pseudodepth"),
            ("📤","Output","Pseudosection ρa sintetik · Tabel datum · Export .dat & .csv"),
        ]):
            with col:
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eaf0;border-radius:10px;padding:16px'>
                <div style='font-size:1.6rem;text-align:center'>{icon}</div>
                <b style='font-size:.88rem;color:#0a2744;display:block;text-align:center;margin:6px 0'>{title}</b>
                <p style='font-size:.8rem;color:#6b7280;text-align:center;margin:0'>{body}</p>
                </div>""", unsafe_allow_html=True)
    else:
        keys   = list(fwd.keys())
        labels = {k: f"a={k.split('_')[0]}m — {k.split('_',1)[1]}" for k in keys}

        sel = st.selectbox("Pilih simulasi", keys, format_func=lambda k: labels[k])
        res = fwd[sel]
        dp  = res.datum_points

        st.markdown(f"""
        <div style='background:#f8fafc;border:1px solid #e8eaf0;border-radius:10px;
          padding:12px 18px;margin-bottom:12px;font-size:.86rem'>
          <b>Konfigurasi:</b> {res.array_name} &nbsp;·&nbsp;
          <b>Spasi a:</b> {res.electrode_spacing} m &nbsp;·&nbsp;
          <b>N-level maks:</b> {res.n_max} &nbsp;·&nbsp;
          <b>Total datum:</b> {len(dp)} &nbsp;·&nbsp;
          <b>ρa range:</b> {dp[:,2].min():.1f} – {dp[:,2].max():.1f} Ω·m
        </div>""", unsafe_allow_html=True)

        all_rho = np.concatenate([r.datum_points[:,2] for r in fwd.values()])
        vlo = float(np.percentile(all_rho,2))
        vhi = float(np.percentile(all_rho,98))

        st.plotly_chart(
            plot_pseudosection(dp,
                title=f"Forward — {res.array_name}, a={res.electrode_spacing}m",
                height=380, rho_min=max(rho_min_inp,vlo),
                rho_max=min(rho_max_inp,vhi), show_values=show_vals),
            use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("ρa min",  f"{dp[:,2].min():.1f} Ω·m")
        c2.metric("ρa maks", f"{dp[:,2].max():.1f} Ω·m")
        c3.metric("ρa rata", f"{dp[:,2].mean():.1f} Ω·m")
        c4.metric("Datum",   len(dp))

        with st.expander("⬇ Export data"):
            df_ex = pd.DataFrame({
                "X_mid (m)": dp[:,0],
                "Pseudo_depth (m)": dp[:,1],
                "Rho_apparent (Ohm.m)": dp[:,2],
                "K_geometric": res.geometric_factors,
            })
            co1,co2 = st.columns(2)
            with co1:
                st.download_button("📄 Download CSV",
                    data=df_ex.to_csv(index=False),
                    file_name=f"fwd_{sel}.csv", mime="text/csv",
                    use_container_width=True)
            with co2:
                code_map = {"Wenner-Schlumberger":"7","Wenner":"1","Dipole-Dipole":"3"}
                lines = [f"Forward_{res.array_name.replace(' ','_')}_a{res.electrode_spacing}m",
                         str(res.electrode_spacing),
                         code_map.get(res.array_name,"7"), "1", str(len(dp))]
                for r in df_ex.values:
                    lines.append(f"  {r[0]:.2f}  {r[1]:.3f}  {r[2]:.3f}")
                st.download_button("📁 Download .dat (RES2DINV format)",
                    data="\n".join(lines),
                    file_name=f"fwd_{sel}.dat", mime="text/plain",
                    use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 4 — KOMPARASI SPASI
# ═══════════════════════════════════════════════════
with tab4:
    if not fwd:
        st.markdown('<div class="info-box">💡 Jalankan forward modelling terlebih dahulu.</div>',
                    unsafe_allow_html=True)
    else:
        avail_arr = sorted(set(k.split("_",1)[1] for k in fwd))
        sel_arr   = st.selectbox("Konfigurasi referensi", avail_arr, key="tab4_arr")
        sp_res    = {float(k.split("_")[0]): v
                     for k,v in fwd.items() if k.split("_",1)[1]==sel_arr}

        if not sp_res:
            st.warning(f"Tidak ada data untuk konfigurasi {sel_arr}.")
        else:
            all_r  = np.concatenate([r.datum_points[:,2] for r in sp_res.values()])
            vlo,vhi = float(np.percentile(all_r,2)), float(np.percentile(all_r,98))

            st.markdown('<div class="sec-head">Perbandingan Pseudosection — Variasi Spasi Elektroda</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sec-sub">Konfigurasi: {sel_arr} · Skala warna identik di semua panel</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_comparison_spasi(sp_res, rho_min=vlo, rho_max=vhi),
                use_container_width=True)

            # Tabel statistik
            st.markdown('<div class="sec-head">Statistik Perbandingan</div>', unsafe_allow_html=True)
            rows = []
            for s,r in sorted(sp_res.items()):
                dp  = r.datum_points
                inv_depth = dp[:,1].max()
                rows.append({
                    "Spasi a (m)": s,
                    "Datum": len(dp),
                    "ρa min (Ω·m)": round(dp[:,2].min(),1),
                    "ρa maks (Ω·m)": round(dp[:,2].max(),1),
                    "ρa rata-rata (Ω·m)": round(dp[:,2].mean(),1),
                    "Kedalaman maks (m)": round(inv_depth,2),
                    "Kategori": "Dangkal" if s<=1 else ("Menengah" if s<=2 else "Dalam"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Analisis teks
            st.markdown('<div class="sec-head">Analisis Pengaruh Spasi</div>', unsafe_allow_html=True)
            icons   = {1:"🔵", 2:"🟢", 3:"🟡", 4:"🟠", 5:"🔴"}
            descs   = {
                1:"Resolusi lateral tinggi, kedalaman terbatas. Baik untuk detail struktur dangkal.",
                2:"Keseimbangan resolusi-kedalaman optimal. Disarankan untuk survei umum. ✓",
                3:"Kedalaman investigasi lebih dalam. Resolusi lateral berkurang.",
                4:"Cocok untuk target geologi dalam. Detail permukaan hilang.",
                5:"Investigasi sangat dalam. Hanya struktur besar yang terdeteksi.",
            }
            for s,r in sorted(sp_res.items()):
                dp = r.datum_points
                ic = icons.get(int(s),"⚪")
                desc = descs.get(int(s), "")
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eaf0;border-radius:8px;
                  padding:10px 14px;margin-bottom:6px;display:flex;gap:12px;align-items:center'>
                  <span style='font-size:1.2rem'>{ic}</span>
                  <div>
                    <b>Spasi a = {s} m</b> &nbsp;·&nbsp;
                    Depth maks ≈ {dp[:,1].max():.1f} m &nbsp;·&nbsp;
                    {len(dp)} datum<br>
                    <span style='font-size:.8rem;color:#6b7280'>{desc}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 5 — KOMPARASI KONFIGURASI
# ═══════════════════════════════════════════════════
with tab5:
    if not fwd:
        st.markdown('<div class="info-box">💡 Jalankan forward modelling terlebih dahulu.</div>',
                    unsafe_allow_html=True)
    else:
        avail_sp = sorted(set(float(k.split("_")[0]) for k in fwd))
        sel_sp   = st.selectbox("Spasi referensi (m)", avail_sp,
                                index=min(1,len(avail_sp)-1), key="tab5_sp")
        arr_res  = {k.split("_",1)[1]: v
                    for k,v in fwd.items() if float(k.split("_")[0])==sel_sp}

        if not arr_res:
            st.warning(f"Tidak ada data untuk spasi {sel_sp} m.")
        else:
            all_r  = np.concatenate([r.datum_points[:,2] for r in arr_res.values()])
            vlo,vhi = float(np.percentile(all_r,2)), float(np.percentile(all_r,98))

            st.markdown('<div class="sec-head">Perbandingan Pseudosection — Variasi Konfigurasi</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sec-sub">Spasi: a = {sel_sp} m · Skala warna identik</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_comparison_array(arr_res, rho_min=vlo, rho_max=vhi),
                use_container_width=True)

            # Tabel
            st.markdown('<div class="sec-head">Statistik Perbandingan Konfigurasi</div>',
                        unsafe_allow_html=True)
            cfg_info = {
                "Wenner-Schlumberger": ("Vertikal + lateral","Sedang","Serbaguna"),
                "Wenner":              ("Vertikal","Tinggi","Lapisan horizontal"),
                "Dipole-Dipole":       ("Lateral","Rendah","Struktur vertikal / patahan"),
            }
            rows = []
            for arr,r in arr_res.items():
                dp   = r.datum_points
                info = cfg_info.get(arr,("-","-","-"))
                rows.append({
                    "Konfigurasi": arr,
                    "Datum": len(dp),
                    "ρa min (Ω·m)": round(dp[:,2].min(),1),
                    "ρa maks (Ω·m)": round(dp[:,2].max(),1),
                    "ρa rata-rata (Ω·m)": round(dp[:,2].mean(),1),
                    "Sensitivitas": info[0],
                    "Rasio S/N": info[1],
                    "Cocok untuk": info[2],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Rec cards
            st.markdown('<div class="sec-head">Rekomendasi Konfigurasi</div>', unsafe_allow_html=True)
            rc1,rc2,rc3 = st.columns(3)
            rec_data = [
                (rc1,"#EFF6FF","#B5D4F4","#0C447C","🔵 Wenner-Schlumberger",
                 "Direkomendasikan untuk survei umum. Sensitif vertikal + lateral, cocok untuk mapping lapisan 2D serbaguna."),
                (rc2,"#E1F5EE","#9FE1CB","#085041","🟢 Wenner",
                 "Ideal untuk lapisan mendatar. Rasio S/N tertinggi — terbaik di area dengan noise lapangan tinggi."),
                (rc3,"#FAECE7","#F5C4B3","#712B13","🔴 Dipole-Dipole",
                 "Optimal mendeteksi patahan dan kontak litologi vertikal. Resolusi lateral terbaik namun sinyal lebih lemah."),
            ]
            for col,bg,border,txt,title,desc in rec_data:
                if title.split(" ",1)[1].replace(" ","").split("—")[0] in arr_res or True:
                    with col:
                        st.markdown(f"""
                        <div style='background:{bg};border:1px solid {border};border-radius:10px;
                          padding:14px;min-height:130px'>
                          <b style='color:{txt}'>{title}</b><br>
                          <span style='font-size:.8rem;color:#374151;line-height:1.55'>{desc}</span>
                        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 6 — LAPORAN & EXPORT
# ═══════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-head">Export & Laporan Penelitian</div>', unsafe_allow_html=True)

    # Ringkasan semua simulasi
    if fwd:
        st.markdown('<div class="sec-sub">Semua simulasi yang telah dijalankan</div>',
                    unsafe_allow_html=True)
        sum_rows = []
        for key,res in sorted(fwd.items()):
            dp = res.datum_points
            sum_rows.append({
                "Simulasi": key,
                "Konfigurasi": res.array_name,
                "Spasi a (m)": res.electrode_spacing,
                "N-level maks": res.n_max,
                "Total datum": len(dp),
                "ρa min (Ω·m)": round(dp[:,2].min(),1),
                "ρa maks (Ω·m)": round(dp[:,2].max(),1),
                "ρa rata-rata (Ω·m)": round(dp[:,2].mean(),1),
                "Depth maks (m)": round(dp[:,1].max(),2),
            })
        df_sum = pd.DataFrame(sum_rows)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="sec-head">Download File</div>', unsafe_allow_html=True)

        co1,co2,co3 = st.columns(3)

        # CSV semua datum
        with co1:
            all_rows = []
            for key,res in sorted(fwd.items()):
                dp = res.datum_points
                for row in dp:
                    all_rows.append({
                        "Simulasi": key,
                        "Konfigurasi": res.array_name,
                        "Spasi_a_m": res.electrode_spacing,
                        "X_mid_m": row[0],
                        "Depth_m": row[1],
                        "Rho_a_Ohmm": round(row[2],2),
                    })
            df_all = pd.DataFrame(all_rows)
            st.download_button(
                "📄 Download semua datum (CSV)",
                data=df_all.to_csv(index=False),
                file_name="geoforward_semua_datum.csv",
                mime="text/csv",
                use_container_width=True,
                help="Semua titik datum dari semua simulasi dalam satu file CSV",
            )

        # CSV ringkasan
        with co2:
            st.download_button(
                "📊 Download ringkasan statistik (CSV)",
                data=df_sum.to_csv(index=False),
                file_name="geoforward_ringkasan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Laporan HTML
        with co3:
            html_report = build_html_report(inv, fwd)
            st.download_button(
                "📄 Download Laporan HTML",
                data=html_report,
                file_name="laporan_geoforward.html",
                mime="text/html",
                use_container_width=True,
                help="Laporan lengkap siap cetak / presentasi",
            )

        st.markdown("""
        <div class="info-box">
        📌 <b>Cara menggunakan file export:</b><br>
        • File <b>.csv</b> dapat dibuka di Excel/Spreadsheet untuk analisis lanjutan<br>
        • File <b>.dat</b> (dari tab Forward Modelling) dapat di-import langsung ke RES2DINV untuk inverse modelling<br>
        • File <b>laporan HTML</b> dapat dibuka di browser dan langsung dicetak (Ctrl+P) atau disimpan sebagai PDF
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="warn-box">⚠️ Jalankan forward modelling terlebih dahulu untuk mengaktifkan fitur export.</div>',
                    unsafe_allow_html=True)

    # Panduan format .dat
    with st.expander("📖 Format file .dat untuk RES2DINV"):
        st.markdown("""
        File `.dat` yang dihasilkan aplikasi ini memiliki format standar RES2DINV:
        ```
        Nama_Simulasi           ← baris 1: nama lintasan
        2.0                     ← baris 2: spasi elektroda (m)
        7                       ← baris 3: kode konfigurasi (1=W, 3=DD, 7=WS)
        1                       ← baris 4: flag
        84                      ← baris 5: jumlah datum
          2.00   0.519   128.40  ← datum: x_mid  pseudo_depth  rho_a
          4.00   0.519   143.20
          ...
        ```
        File ini dapat langsung dibuka di RES2DINV untuk proses inverse modelling lanjutan.
        """)

    # Referensi
    st.markdown("---")
    st.markdown('<div class="sec-head">Referensi Ilmiah</div>', unsafe_allow_html=True)
    refs = [
        ("Loke, M.H. (2004)","*2D and 3D electrical imaging surveys*. Geotomo Software Manual."),
        ("Loke, M.H. (2004)","*RES2DMOD ver. 3.01 — Rapid 2D Resistivity Forward Modelling*. Geotomo Software."),
        ("Telford, Geldart & Sheriff (1990)","*Applied Geophysics* (2nd ed.). Cambridge University Press."),
        ("Reynolds, J.M. (1997)","*An Introduction to Applied and Environmental Geophysics*. Wiley."),
        ("Geyer, R. (1983)","*Sensitivity analysis for electrical resistivity surveys*. Geophysics 48(7)."),
    ]
    for author, title in refs:
        st.markdown(f"- **{author}** — {title}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🌍 <b>GeoForward 2D</b> &nbsp;·&nbsp;
  Aplikasi Forward Modelling Geolistrik Tahanan Jenis 2D &nbsp;·&nbsp;
  Berbasis <a href='https://streamlit.io' target='_blank'>Streamlit</a> &nbsp;·&nbsp;
  Untuk keperluan penelitian & akademik
</div>
""", unsafe_allow_html=True)
