
# =============================================================================
#  APLIKASI ANALISIS GRAF - UAS MATEMATIKA DISKRIT
#  File   : app_graf_uas.py
#  Deskripsi : Aplikasi Streamlit untuk memvisualisasikan dan menganalisis
#              Graf Ganda Tak Berarah (Undirected Multigraph) secara manual.
# =============================================================================

# --- Import Library ---
# Streamlit  : framework web app Python, bikin UI interaktif tanpa HTML manual.
# NetworkX   : library khusus teori graf — bikin graph, hitung derajat, layout, dll.
# Matplotlib : library plotting Python, dipakai buat gambar visualisasi graf.
# patches    : sub-modul Matplotlib buat gambar bentuk geometri (lingkaran gelang).
# NumPy      : library komputasi numerik, dipakai buat bikin matriks (array 2D).
# Pandas     : library data tabular, buat bikin DataFrame (tabel matriks).
# defaultdict & Counter : utility dari collections —
#   defaultdict: dict yang auto-inisialisasi value,
#   Counter: hitung frekuensi elemen (dipakai hitung sisi ganda).
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# -----------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# -----------------------------------------------------------
# set_page_config() WAJIB dipanggil pertama kali sebelum komponen lain.
# Fungsinya mengatur metadata halaman: judul tab browser, ikon tab,
# dan layout "wide" supaya halaman memanfaatkan seluruh lebar layar
# (cocok karena kita pakai 2 kolom: input + visualisasi).
st.set_page_config(
    page_title="Analisis Graf - UAS Matematika Diskrit",
    page_icon="🔷",
    layout="wide"
)

# -----------------------------------------------------------
# JUDUL UTAMA APLIKASI
# -----------------------------------------------------------
# st.title() -> heading besar di atas halaman.
# st.caption() -> teks kecil abu-abu sebagai subtitle / keterangan singkat.
st.title("Graf")
st.caption("Aplikasi untuk UAS Matematika Diskrit | Input Manual | Graf G = (V, E)")

st.markdown("---")

# =============================================================================
#  FUNGSI-FUNGSI PEMBANTU (HELPER FUNCTIONS)
# =============================================================================

# - Fungsi ini mengubah teks input jadi list simpul.
# - Mendukung dua format: koma ("a, b, c") dan spasi ("a b c").
# - Duplikat otomatis dihapus karena dalam teori graf,
#   himpunan simpul V tidak boleh ada elemen ganda.
#
# "Kenapa duplikat dihapus?"
#
# Jawaban:
# Karena V adalah himpunan (set), dan sifat himpunan
# adalah setiap elemen harus unik. Jadi kalau user
# menulis "a, b, a", kita hanya simpan {a, b}.
# ===============================
def parse_simpul(teks_simpul: str) -> list:
    """
    Memparse teks input simpul/titik menjadi sebuah list.
    Contoh input: "a, b, c, d" atau "a b c d"
    Contoh output: ['a', 'b', 'c', 'd']
    """
    if not teks_simpul.strip():
        return []
    # replace(",", " ") -> normalisasi delimiter: koma dijadikan spasi,
    # supaya user bebas pakai koma atau spasi. Lalu split() pecah per kata.
    # Ganti koma dengan spasi lalu pecah berdasarkan spasi
    token = teks_simpul.replace(",", " ").split()
    # Buang duplikat tapi pertahankan urutan
    # Pakai set "seen" untuk tracking, bukan langsung set(token),
    # karena set() tidak menjaga urutan input user.
    seen = set()
    simpul_unik = []
    for t in token:
        if t not in seen:
            seen.add(t)
            simpul_unik.append(t)
    return simpul_unik


# ===============================
# Tips Presentasi — parse_sisi
# -------------------------------
# Yang bisa dijelaskan ke dosen:
#
# - Setiap baris = satu sisi (edge) dari graf.
# - Format per baris: "u v" atau "u, v"
# - Kalau u == v (misal "b b"), itu berarti gelang/loop.
# - Kalau pasangan sama ditulis berulang, itu sisi ganda (multi-edge).
# - Baris yang cuma punya 1 token diabaikan (bukan sisi valid).
#
# Jika dosen bertanya "Kenapa input per baris, bukan satu baris semua?"
#
# Jawaban:
# Karena setiap sisi punya dua endpoint. Kalau dijadikan satu baris,
# parsing jadi ambigu — mana pasangan mana. Per baris = satu edge,
# jauh lebih jelas dan mengurangi error input.
# ===============================
def parse_sisi(teks_sisi: str) -> list:
    """
    Memparse teks input sisi/garis menjadi list of tuples.
    Setiap baris berisi satu pasang simpul.
    Contoh input (multiline):
        a b
        b c
        c d
        c d
        b b
    Contoh output: [('a','b'), ('b','c'), ('c','d'), ('c','d'), ('b','b')]
    """
    sisi_list = []
    if not teks_sisi.strip():
        return sisi_list
    # Iterasi per baris — setiap baris merepresentasikan satu sisi e = (u, v)
    for baris in teks_sisi.strip().split("\n"):
        baris = baris.strip()
        if not baris:
            continue
        # Ganti koma dengan spasi untuk fleksibilitas input
        token = baris.replace(",", " ").split()
        # Minimal harus ada 2 token untuk membentuk pasangan (u, v)
        if len(token) >= 2:
            sisi_list.append((token[0], token[1]))
        # Jika hanya satu token, abaikan (input tidak valid)
    return sisi_list


# ===============================
# Tips Presentasi — hitung_derajat
# -------------------------------
# Yang bisa dijelaskan ke dosen:
#
# - Derajat (degree) simpul v, ditulis d(v), adalah
#   jumlah sisi yang terhubung ke simpul v.
# - Aturan khusus: gelang/loop dihitung 2, bukan 1.
#   Ini karena gelang "menyentuh" simpul tersebut dua kali
#   (kedua ujungnya ada di simpul yang sama).
# - Ini sesuai dengan definisi formal:
#   d(v) = jumlah sisi yang incident dengan v,
#   dimana loop dihitung dua kali.
#
# Jika dosen bertanya "Kenapa loop dihitung 2?"
#
# Jawaban:
# Karena definisi derajat menghitung jumlah "ujung sisi"
# yang menyentuh simpul. Loop punya 2 ujung, keduanya
# menyentuh simpul yang sama, jadi d(v) += 2.
# Ini juga yang menjamin Handshaking Theorem tetap berlaku:
# Σ d(v) = 2|E|.
# ===============================
def hitung_derajat(G: nx.MultiGraph, simpul_list: list) -> dict:
    """
    Menghitung derajat setiap simpul secara manual sesuai aturan:
    - Sisi biasa (u, v) menambah 1 pada derajat u dan 1 pada derajat v.
    - Gelang/Loop (u, u) menambah 2 pada derajat u (dihitung dua kali).
    Mengembalikan dictionary {simpul: derajat}.
    """
    # Inisialisasi semua simpul dengan derajat 0
    derajat = {v: 0 for v in simpul_list}

    # Iterasi semua sisi di MultiGraph (termasuk sisi ganda dan gelang)
    # G.edges(keys=True) mengembalikan (u, v, key) —
    # key adalah ID unik untuk sisi ganda antara u dan v.
    for u, v, _ in G.edges(data=False, keys=True):
        if u == v:
            # Gelang/Loop: tambahkan 2 ke simpul yang sama
            if u in derajat:
                derajat[u] += 2
        else:
            # Sisi biasa: masing-masing simpul mendapat +1
            if u in derajat:
                derajat[u] += 1
            if v in derajat:
                derajat[v] += 1

    return derajat


# ===============================
# Tips Presentasi — Matriks Ketetanggaan
# -------------------------------
# Yang bisa dijelaskan ke dosen:
#
# - Adjacency Matrix A berukuran |V| × |V|.
# - A[i][j] = jumlah sisi antara simpul i dan simpul j.
# - Karena graf tak berarah, matriks ini SIMETRIS: A[i][j] == A[j][i].
# - Untuk gelang (loop), nilainya ada di diagonal: A[i][i].
#
# Jika dosen bertanya "Kenapa matriks simetris?"
#
# Jawaban:
# Karena graf tak berarah, sisi {u,v} = sisi {v,u}.
# Jadi kalau A[i][j] = 1, otomatis A[j][i] juga harus 1.
# ===============================
def buat_matriks_ketetanggaan(G: nx.MultiGraph, simpul_list: list) -> pd.DataFrame:
    """
    Membuat Matriks Ketetanggaan (Adjacency Matrix) ukuran V x V.
    Nilai A[i][j] = jumlah sisi antara simpul i dan simpul j.
    Untuk gelang/loop (i == i), nilai diisi dengan 1 (konvensi umum dalam kuliah).
    """
    n = len(simpul_list)
    # Buat matriks kosong berukuran n x n
    # np.zeros -> semua elemen awal = 0 (belum ada sisi)
    matriks = np.zeros((n, n), dtype=int)

    # Peta simpul -> indeks numerik, supaya bisa akses matriks[i][j]
    idx = {v: i for i, v in enumerate(simpul_list)}  # peta simpul -> indeks

    for u, v, _ in G.edges(data=False, keys=True):
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        if u == v:
            # Gelang: tambah 1 pada diagonal (konvensi matriks ketetanggaan)
            matriks[i][i] += 1
        else:
            # Sisi tak berarah: isi kedua posisi (simetris)
            matriks[i][j] += 1
            matriks[j][i] += 1  # Graf tak berarah → matriks simetris

    # Bungkus ke DataFrame supaya tampil sebagai tabel berlabel
    df = pd.DataFrame(matriks, index=simpul_list, columns=simpul_list)
    return df


# ===============================
# Tips Presentasi — Matriks Bersisian
# -------------------------------
# Yang bisa dijelaskan ke dosen:
#
# - Incidence Matrix B berukuran |V| × |E|.
#   Baris = simpul, Kolom = sisi.
# - B[i][k] = 1 jika simpul i adalah endpoint dari sisi ek.
# - B[i][k] = 2 jika sisi ek adalah gelang di simpul i.
#   Angka 2 karena gelang "incident" dua kali ke simpul yang sama.
# - Setiap kolom (sisi biasa) pasti punya tepat dua angka 1
#   (di baris kedua endpoint-nya).
#
# Jika dosen bertanya "Kenapa gelang bernilai 2, bukan 1?"
#
# Jawaban:
# Konvensi incidence matrix: jumlah incident per simpul.
# Gelang punya kedua ujung di simpul yang sama,
# jadi incident-nya 2. Ini juga menjaga konsistensi
# bahwa jumlah setiap kolom selalu = 2.
# ===============================
def buat_matriks_bersisian(G: nx.MultiGraph, simpul_list: list, sisi_list: list) -> pd.DataFrame:
    """
    Membuat Matriks Bersisian (Incidence Matrix) ukuran V x E.
    Setiap sisi mendapat kolom sendiri, termasuk sisi ganda dan gelang.
    - Sisi biasa (u,v): nilai 1 pada baris u dan baris v di kolom sisi tersebut.
    - Gelang (u,u): nilai 2 pada baris u di kolom sisi tersebut.
    """
    n_simpul = len(simpul_list)
    n_sisi = len(sisi_list)

    # Buat matriks kosong
    matriks = np.zeros((n_simpul, n_sisi), dtype=int)

    idx = {v: i for i, v in enumerate(simpul_list)}  # peta simpul -> indeks

    # Buat label kolom untuk setiap sisi: e1, e2, e3, ...
    # Label ini akan jadi header kolom di tabel output.
    label_sisi = [f"e{k+1}" for k in range(n_sisi)]

    for k, (u, v) in enumerate(sisi_list):
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        if u == v:
            # Gelang: nilai 2 pada baris simpul tersebut
            matriks[i][k] = 2
        else:
            # Sisi biasa: nilai 1 pada baris kedua simpul
            matriks[i][k] = 1
            matriks[j][k] = 1

    df = pd.DataFrame(matriks, index=simpul_list, columns=label_sisi)
    return df


# ===============================
# Tips Presentasi — gambar_graf
# -------------------------------
# Yang bisa dijelaskan ke dosen:
#
# - Fungsi ini menggambar graf secara visual menggunakan Matplotlib.
# - Menangani 3 kasus: sisi biasa, sisi ganda, dan gelang.
# - spring_layout menghitung posisi node pakai simulasi "pegas"
#   (force-directed), supaya node yang terhubung dekat,
#   yang tidak terhubung menjauh — hasilnya graf enak dilihat.
# - seed=42 supaya posisi node konsisten setiap kali render
#   (tidak berubah-ubah acak).
#
# Jika dosen bertanya "Bagaimana sisi ganda digambar?"
#
# Jawaban:
# Sisi ganda digambar melengkung (arc) dengan sudut berbeda-beda.
# Pakai connectionstyle="arc3, rad=X" dari Matplotlib.
# Nilai rad positif = lengkung ke satu sisi, negatif = sisi lain.
# Semakin banyak sisi ganda, semakin besar radius lengkungnya
# supaya tidak saling tumpuk.
#
# Jika dosen bertanya "Bagaimana gelang digambar?"
#
# Jawaban:
# Gelang digambar sebagai lingkaran kecil (patches.Circle)
# yang diletakkan di luar node, menjauh dari pusat graf.
# Arahnya dihitung dari posisi node ke pusat graf,
# lalu dibalik supaya lingkaran muncul di sisi luar.
# ===============================
def gambar_graf(G: nx.MultiGraph, simpul_list: list, E: list):
    """
    Fungsi utama untuk menggambar graf menggunakan Matplotlib.
    Menangani sisi biasa, sisi ganda (multi-edge), dan gelang (loop).
    """
    # --- SETUP FIGURE ---
    # figsize=(4, 3) -> ukuran canvas gambar dalam inci (sengaja kecil)
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')  # Background area graf sedikit abu-abu
    ax.set_aspect('equal')       # Rasio x:y sama supaya lingkaran tetap bulat
    ax.axis('off')               # Hilangkan sumbu x dan y (tidak relevan untuk graf)

    # Jika tidak ada simpul, tampilkan pesan kosong
    if len(simpul_list) == 0:
        ax.text(0.5, 0.5, "Belum ada simpul untuk ditampilkan.",
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        return fig

    # --- LAYOUT: spring_layout dengan seed konsisten ---
    # spring_layout -> algoritma force-directed (Fruchterman-Reingold).
    # k=1.5 -> jarak ideal antar-node (lebih besar = lebih renggang).
    # seed=42 -> posisi tetap sama setiap kali di-render (reproducible).
    pos = nx.spring_layout(G, k=1.5, seed=42)

    # Simpul terisolasi (tidak punya sisi) tidak masuk layout NetworkX,
    # jadi kita beri posisi acak supaya tetap muncul di gambar.
    for v in simpul_list:
        if v not in pos:
            pos[v] = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

    # --- EXACT DRAWING LOGIC TO COPY-PASTE ---
    # 1. Gambar Simpul (Nodes) dan Label
    # node_color='lightblue' -> warna isi node
    # edgecolors='midnightblue' -> warna garis tepi node
    # node_size=800 -> ukuran node, font_weight='bold' -> label tebal
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', edgecolors='midnightblue', node_size=800, linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold', font_color='midnightblue')

    # 2. Hitung frekuensi sisi tak berarah
    # Counter menghitung berapa kali setiap pasangan simpul muncul.
    # sorted((u,v)) supaya (a,b) dan (b,a) dianggap sama (tak berarah).
    # Ini diperlukan untuk tahu mana yang sisi tunggal, mana sisi ganda.
    import collections
    edge_counts = collections.Counter([tuple(sorted((u, v))) for u, v in E if u != v])
    drawn_counts = {pair: 0 for pair in edge_counts}

    # Pusat graf untuk arah gelang (loop)
    # Hitung titik tengah semua node — dipakai untuk menentukan
    # arah gelang supaya lingkaran muncul ke sisi luar graf.
    center_x = sum(p[0] for p in pos.values()) / len(pos) if pos else 0
    center_y = sum(p[1] for p in pos.values()) / len(pos) if pos else 0

    # 3. Gambar Sisi (Edges) dan Gelang (Loops)
    import matplotlib.patches as patches
    import numpy as np

    for u, v in E:
        if u == v: # LOGIKA GELANG (LOOP)
            x, y = pos[u]
            # Hitung arah dari pusat graf ke node ini
            dx, dy = x - center_x, y - center_y
            dist = np.hypot(dx, dy)  # jarak = sqrt(dx² + dy²)
            if dist == 0: dx, dy, dist = 0, 1, 1 # Default jika di tengah persis

            # Dorong posisi gelang ke luar dari pusat graf
            # supaya lingkaran gelang tidak menimpa node lain.
            loop_x = x + (dx / dist) * 0.15
            loop_y = y + (dy / dist) * 0.15

            # patches.Circle -> gambar lingkaran kecil (gelang)
            # fill=False -> tidak diisi warna (outline saja)
            # zorder=0 -> gambar di belakang node supaya tidak menutupi
            circle = patches.Circle((loop_x, loop_y), radius=0.15, fill=False, color='midnightblue', linewidth=2, zorder=0)
            ax.add_patch(circle)

        else: # LOGIKA SISI & SISI GANDA (MULTIPLE EDGES)
            pair = tuple(sorted((u, v)))
            total = edge_counts[pair]     # total sisi antara u dan v
            current = drawn_counts[pair]  # sudah digambar berapa

            if total == 1:
                # Sisi tunggal: gambar garis lurus biasa
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color='darkgray', width=2)
            else:
                # Sisi ganda: gambar garis melengkung (arc)
                # sign bergantian +/- supaya lengkung ke kiri dan kanan
                # step makin besar supaya sisi ke-3, ke-4 dst makin melengkung
                sign = 1 if current % 2 == 0 else -1
                step = (current + 1) // 2
                rad = 0.2 * sign * step  # radius lengkungan
                # connectionstyle="arc3, rad=X" -> garis melengkung Matplotlib
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, connectionstyle=f"arc3, rad={rad}", edge_color='darkgray', width=2)

            drawn_counts[pair] += 1
    # ------------------------------------------

    # Atur batas tampilan agar tidak terpotong
    ax.margins(0.25)
    plt.tight_layout()

    return fig


# =============================================================================
#  TATA LETAK HALAMAN: 2 KOLOM UTAMA
#  col1 = Input (lebih sempit), col2 = Visualisasi + Analisis (lebih lebar)
# =============================================================================
# st.columns([1, 2.5]) membagi halaman jadi 2 kolom dengan rasio 1:2.5.
# Kolom kiri (sempit) untuk input, kolom kanan (lebar) untuk output.
# Ini memanfaatkan layout="wide" yang sudah diset di awal.
col1, col2 = st.columns([1, 2.5])

# -----------------------------------------------------------
# KOLOM 1: PANEL INPUT DATA
# -----------------------------------------------------------
with col1:
    st.subheader("Input Data Graf")
    st.markdown("---")

    # INPUT 1: Himpunan Simpul
    # text_input -> satu baris input teks, cocok untuk daftar simpul
    # label_visibility="collapsed" -> sembunyikan label (sudah ada markdown di atas)
    st.markdown("**Himpunan Simpul/Titik (V):**")
    teks_simpul = st.text_input(
        label="Masukkan Himpunan Simpul/Titik (V):",
        value="",
        placeholder="Contoh: a, b, c, d",
        help="Masukkan nama simpul dipisahkan koma atau spasi. Contoh: a, b, c, d",
        label_visibility="collapsed"
    )
    # caption -> teks petunjuk kecil di bawah input
    st.caption("💡 Pisahkan dengan koma atau spasi. Contoh: `a, b, c, d`")

    st.markdown("&nbsp;", unsafe_allow_html=True)  # Spasi vertikal

    # INPUT 2: Himpunan Sisi
    # text_area -> input multi-baris, cocok karena setiap sisi = satu baris
    st.markdown("**Himpunan Sisi/Garis (E):**")
    teks_sisi = st.text_area(
        label="Masukkan Himpunan Sisi/Garis (E):",
        value="",
        height=180,
        placeholder="Satu pasang simpul per baris.\nContoh:\na b\nb c\nc d",
        help="Masukkan satu pasang simpul per baris. Gelang/Loop: tulis simpul yang sama dua kali (misal: b b)",
        label_visibility="collapsed"
    )
    st.caption("💡 Satu pasang per baris.\n🔁 Gelang: tulis simpul yang sama (misal: `b b`)\n↔️ Sisi ganda: tulis pasang yang sama berulang")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # TOMBOL PROSES
    # type="primary" -> tombol berwarna (menonjol, bukan abu-abu)
    # width='stretch' -> tombol selebar kolom
    tombol_proses = st.button(
        "🔍 Analisis Graf",
        type="primary",
        width='stretch'
    )

    st.markdown("---")

    # BOX PETUNJUK PENGGUNAAN
    # expander -> panel collapsible, expanded=False supaya default tertutup
    # (tidak memenuhi layar, user buka kalau perlu)
    with st.expander("📖 Petunjuk Penggunaan", expanded=False):
        st.markdown("""
        **Cara Input:**
        1. Masukkan nama simpul di field pertama, pisahkan dengan koma.
        2. Masukkan sisi di field kedua — satu pasang simpul per baris.
        3. Klik tombol **Analisis Graf**.

        **Contoh Input:**
        ```
        Simpul: a, b, c, d
        Sisi:
        a b   ← sisi biasa
        b c   ← sisi biasa
        c d   ← sisi ganda (ditulis 2x)
        c d   ← sisi ganda
        b b   ← gelang/loop di simpul b
        ```

        **Terminologi:**
        - **Simpul/Titik (V)** = Vertex/Node
        - **Sisi/Garis (E)** = Edge
        - **Gelang/Loop** = Self-loop (u, u)
        - **Sisi Ganda** = Multiple Edge (u,v) > 1x
        - **Derajat (d)** = Degree
        """)

# -----------------------------------------------------------
# KOLOM 2: VISUALISASI DAN ANALISIS
# -----------------------------------------------------------
with col2:
    # Proses data saat tombol diklik atau saat pertama kali load
    # Auto-render: selama ada input simpul, langsung tampilkan hasilnya
    # (tidak harus klik tombol — lebih responsif)
    if teks_simpul.strip():  # Auto-render jika ada input

        # --- PARSING INPUT ---
        simpul_list = parse_simpul(teks_simpul)
        sisi_list = parse_sisi(teks_sisi)

        # Validasi: pastikan semua simpul pada sisi ada di daftar simpul V.
        # Ini penting karena user bisa saja mengetik sisi "x y"
        # padahal simpul x atau y tidak ada di V.
        simpul_set = set(simpul_list)
        sisi_valid = []
        sisi_tidak_valid = []
        for u, v in sisi_list:
            if u in simpul_set and v in simpul_set:
                sisi_valid.append((u, v))
            else:
                sisi_tidak_valid.append((u, v))

        # Tampilkan peringatan untuk sisi tidak valid
        if sisi_tidak_valid:
            st.warning(
                f"⚠️ Sisi berikut diabaikan karena simpulnya tidak ada di V: "
                f"{sisi_tidak_valid}"
            )

        # --- BANGUN MULTIGRAPH ---
        # Membuat objek MultiGraph dari NetworkX.
        # MultiGraph dipilih karena aplikasi ini menganalisis
        # graf ganda tak berarah sehingga mendukung
        # multiple edge (sisi ganda) dan loop (gelang).
        # Kalau pakai Graph() biasa, sisi ganda dan loop tidak bisa disimpan.
        G = nx.MultiGraph()
        # add_nodes_from() menambahkan semua simpul sekaligus dari list,
        # termasuk simpul terisolasi (yang tidak punya sisi).
        G.add_nodes_from(simpul_list)  # Tambahkan semua simpul (termasuk terisolasi)
        for u, v in sisi_valid:
            G.add_edge(u, v)           # MultiGraph otomatis izinkan duplikat dan loop

        # --- INFO RINGKAS GRAF ---
        # Tampilkan 3 metrik utama dalam 3 kolom sejajar
        # metric() menampilkan angka besar dengan label — cocok untuk dashboard
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Jumlah Simpul |V|", len(simpul_list))
        col_info2.metric("Jumlah Sisi |E|", len(sisi_valid))
        col_info3.metric("Jenis Graf", "Graf Ganda Tak Berarah")

        st.markdown("---")

        # --- VISUALISASI GRAF ---
        st.subheader("🖼️ Visualisasi Graf G = (V, E)")
        fig = gambar_graf(G, simpul_list, sisi_valid)
        # Paksa grafik tampil kecil (~1/4 halaman) dengan membungkus dalam kolom sempit
        col_graph, col_spacer = st.columns([1, 1])
        with col_graph:
            # use_container_width=False -> gambar tidak di-stretch ke lebar kolom,
            # ukurannya sesuai figsize yang sudah diset (4x3 inci).
            st.pyplot(fig, use_container_width=False)  # False = tidak stretch ke lebar penuh
        # plt.close() wajib dipanggil setelah st.pyplot()
        # supaya figure Matplotlib dibebaskan dari memori (mencegah memory leak).
        plt.close(fig)  # Tutup figure agar tidak memory leak

        # --- NOTASI FORMAL ---
        # Tampilkan himpunan V dan E dalam notasi matematika formal.
        # Ini penting karena dalam teori graf, graf G didefinisikan sebagai
        # pasangan terurut G = (V, E) dimana V = himpunan simpul, E = himpunan sisi.
        st.markdown("---")
        st.subheader("📐 Notasi Formal Graf")

        # Tampilkan V dan E dalam format himpunan
        v_str = "V = {" + ", ".join(simpul_list) + "}"
        e_str = "E = {" + ", ".join([f"({u},{v})" for u, v in sisi_valid]) + "}"
        st.code(v_str + "\n" + e_str, language=None)

        # --- DERAJAT SIMPUL ---
        st.markdown("---")
        st.subheader("📊 Derajat Simpul d(v)")

        derajat = hitung_derajat(G, simpul_list)
        total_derajat = sum(derajat.values())

        # Tampilkan derajat setiap simpul dalam bentuk tabel (DataFrame).
        # Tabel lebih mudah dibaca daripada teks biasa,
        # apalagi kalau simpulnya banyak.
        df_derajat = pd.DataFrame(
            [(v, derajat[v]) for v in simpul_list],
            columns=["Simpul (v)", "Derajat d(v)"]
        )

        # Tampilkan tabel derajat — paksa center via HTML (bypass Streamlit alignment)
        # Streamlit default tabel rata kiri, pakai HTML styling supaya center.
        html_derajat = (
            df_derajat.style
            .hide(axis='index')
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            .to_html()
        )
        st.markdown(html_derajat, unsafe_allow_html=True)

        # Tandai gelang jika ada (untuk keterangan teks saja)
        simpul_ada_gelang = set()
        for u, v in sisi_valid:
            if u == v:
                simpul_ada_gelang.add(u)

        # ===============================
        # Tips Presentasi — Handshaking Theorem
        # -------------------------------
        # Yang bisa dijelaskan ke dosen:
        #
        # - Handshaking Theorem (Teorema Jabat Tangan):
        #   "Jumlah semua derajat simpul selalu sama dengan
        #    dua kali jumlah sisi."
        #   Σ d(v) = 2|E|
        #
        # - Ini berlaku untuk SEMUA graf, termasuk multigraph.
        # - Intuisinya: setiap sisi punya 2 ujung,
        #   jadi setiap sisi "menyumbang" total 2 ke derajat.
        # - Gelang juga mengikuti aturan ini karena kita
        #   menghitung d(v) += 2 untuk loop.
        #
        # Jika dosen bertanya "Kenapa teorema ini selalu benar?"
        #
        # Jawaban:
        # Karena setiap sisi e = {u,v} menambah 1 ke d(u) dan 1 ke d(v).
        # Totalnya: setiap sisi menyumbang 2 ke total derajat.
        # Maka Σ d(v) = 2 × jumlah sisi = 2|E|.
        # Untuk gelang, satu sisi menambah 2 ke d(u) — tetap sesuai.
        # ===============================

        # Tampilkan total derajat dan verifikasi teorema握手
        col_deg1, col_deg2 = st.columns(2)
        col_deg1.metric(
            "Total Derajat d(G)",
            total_derajat,
            help="Jumlah semua derajat simpul = 2 × |E|"
        )
        col_deg2.metric(
            "2 × |E|",
            2 * len(sisi_valid),
            help="Teorema握手: Total derajat selalu = 2 × jumlah sisi"
        )

        # Verifikasi Teorema握手 (Handshaking Theorem)
        if total_derajat == 2 * len(sisi_valid):
            st.success(
                f"✅ **Teorema握手(Handshaking Theorem) Terpenuhi:** "
                f"Σd(v) = {total_derajat} = 2 × {len(sisi_valid)} = 2|E|"
            )
        else:
            st.error("❌ Ada ketidaksesuaian dalam perhitungan derajat.")

        if simpul_ada_gelang:
            st.info(
                f"🔁 Simpul dengan gelang: {list(simpul_ada_gelang)} — "
                "Gelang menambahkan 2 pada derajat simpul tersebut."
            )

        # --- MATRIKS KETETANGGAAN ---
        st.markdown("---")
        st.subheader("🔢 Matriks Ketetanggaan (Adjacency Matrix) A")
        st.caption(
            "A[i][j] = jumlah sisi antara simpul i dan simpul j. "
            "Untuk gelang, nilai diagonal = jumlah gelang di simpul tersebut."
        )

        df_adj = buat_matriks_ketetanggaan(G, simpul_list)

        # Tampilkan matriks ketetanggaan — paksa center via HTML
        html_adj = (
            df_adj.style
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            .to_html()
        )
        st.markdown(html_adj, unsafe_allow_html=True)

        # --- MATRIKS BERSISIAN ---
        st.markdown("---")
        st.subheader("🔢 Matriks Bersisian (Incidence Matrix) B")
        st.caption(
            "B[i][k] = 1 jika simpul i bersisian dengan sisi ek (sisi biasa). "
            "B[i][k] = 2 jika sisi ek adalah gelang di simpul i."
        )

        if len(sisi_valid) > 0:
            df_inc = buat_matriks_bersisian(G, simpul_list, sisi_valid)
            # Tampilkan matriks bersisian — paksa center via HTML
            html_inc = (
                df_inc.style
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                .to_html()
            )
            st.markdown(html_inc, unsafe_allow_html=True)

            # Keterangan setiap sisi: menampilkan label e1, e2, ... beserta pasangan simpulnya
            st.markdown("**Keterangan Sisi:**")
            keterangan_sisi = []
            for k, (u, v) in enumerate(sisi_valid):
                if u == v:
                    ket = f"e{k+1}: ({u}, {v}) — **Gelang/Loop**"
                else:
                    ket = f"e{k+1}: ({u}, {v})"
                keterangan_sisi.append(ket)

            st.markdown(" | ".join(keterangan_sisi))
        else:
            st.info("Belum ada sisi untuk membuat Matriks Bersisian.")

        st.markdown("---")
        st.caption(
            "📌 Graf Ganda Tak Berarah (Undirected Multigraph) G = (V, E) | "
            "Dibuat dengan NetworkX & Matplotlib | UAS Matematika Diskrit"
        )

    else:
        # Blank start state: tampilkan pesan jika input simpul kosong
        st.info("Silakan masukkan himpunan simpul dan sisi untuk melihat graf.")
