
# =============================================================================
#  APLIKASI ANALISIS GRAF - UAS MATEMATIKA DISKRIT
#  File   : app_graf_uas.py
#  Deskripsi : Aplikasi Streamlit untuk memvisualisasikan dan menganalisis
#              Graf Ganda Tak Berarah (Undirected Multigraph) secara manual.
# =============================================================================

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
st.set_page_config(
    page_title="Analisis Graf - UAS Matematika Diskrit",
    page_icon="🔷",
    layout="wide"
)

# -----------------------------------------------------------
# JUDUL UTAMA APLIKASI
# -----------------------------------------------------------
st.title("Graf")
st.caption("Aplikasi untuk UAS Matematika Diskrit | Input Manual | Graf G = (V, E)")

st.markdown("---")

# =============================================================================
#  FUNGSI-FUNGSI PEMBANTU (HELPER FUNCTIONS)
# =============================================================================

def parse_simpul(teks_simpul: str) -> list:
    """
    Memparse teks input simpul/titik menjadi sebuah list.
    Contoh input: "a, b, c, d" atau "a b c d"
    Contoh output: ['a', 'b', 'c', 'd']
    """
    if not teks_simpul.strip():
        return []
    # Ganti koma dengan spasi lalu pecah berdasarkan spasi
    token = teks_simpul.replace(",", " ").split()
    # Buang duplikat tapi pertahankan urutan
    seen = set()
    simpul_unik = []
    for t in token:
        if t not in seen:
            seen.add(t)
            simpul_unik.append(t)
    return simpul_unik


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
    for baris in teks_sisi.strip().split("\n"):
        baris = baris.strip()
        if not baris:
            continue
        # Ganti koma dengan spasi untuk fleksibilitas input
        token = baris.replace(",", " ").split()
        if len(token) >= 2:
            sisi_list.append((token[0], token[1]))
        # Jika hanya satu token, abaikan (input tidak valid)
    return sisi_list


def hitung_derajat(G: nx.MultiGraph, simpul_list: list) -> dict:
    """
    Menghitung derajat setiap simpul secara manual sesuai aturan:
    - Sisi biasa (u, v) menambah 1 pada derajat u dan 1 pada derajat v.
    - Gelang/Loop (u, u) menambah 2 pada derajat u (dihitung dua kali).
    Mengembalikan dictionary {simpul: derajat}.
    """
    derajat = {v: 0 for v in simpul_list}

    # Iterasi semua sisi di MultiGraph (termasuk sisi ganda dan gelang)
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


def buat_matriks_ketetanggaan(G: nx.MultiGraph, simpul_list: list) -> pd.DataFrame:
    """
    Membuat Matriks Ketetanggaan (Adjacency Matrix) ukuran V x V.
    Nilai A[i][j] = jumlah sisi antara simpul i dan simpul j.
    Untuk gelang/loop (i == i), nilai diisi dengan 1 (konvensi umum dalam kuliah).
    """
    n = len(simpul_list)
    # Buat matriks kosong berukuran n x n
    matriks = np.zeros((n, n), dtype=int)

    idx = {v: i for i, v in enumerate(simpul_list)}  # peta simpul -> indeks

    for u, v, _ in G.edges(data=False, keys=True):
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        if u == v:
            # Gelang: tambah 1 pada diagonal (konvensi matriks ketetanggaan)
            matriks[i][i] += 1
        else:
            matriks[i][j] += 1
            matriks[j][i] += 1  # Graf tak berarah → matriks simetris

    df = pd.DataFrame(matriks, index=simpul_list, columns=simpul_list)
    return df


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


def gambar_graf(G: nx.MultiGraph, simpul_list: list, E: list):
    """
    Fungsi utama untuk menggambar graf menggunakan Matplotlib.
    Menangani sisi biasa, sisi ganda (multi-edge), dan gelang (loop).
    """
    # --- SETUP FIGURE ---
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    ax.set_aspect('equal')
    ax.axis('off')

    # Jika tidak ada simpul, tampilkan pesan kosong
    if len(simpul_list) == 0:
        ax.text(0.5, 0.5, "Belum ada simpul untuk ditampilkan.",
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        return fig

    # --- LAYOUT: spring_layout dengan seed konsisten ---
    pos = nx.spring_layout(G, k=1.5, seed=42)

    # Pastikan simpul terisolasi tetap punya posisi
    for v in simpul_list:
        if v not in pos:
            pos[v] = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

    # --- EXACT DRAWING LOGIC TO COPY-PASTE ---
    # 1. Gambar Simpul (Nodes) dan Label
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', edgecolors='midnightblue', node_size=800, linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold', font_color='midnightblue')

    # 2. Hitung frekuensi sisi tak berarah
    import collections
    edge_counts = collections.Counter([tuple(sorted((u, v))) for u, v in E if u != v])
    drawn_counts = {pair: 0 for pair in edge_counts}

    # Pusat graf untuk arah gelang (loop)
    center_x = sum(p[0] for p in pos.values()) / len(pos) if pos else 0
    center_y = sum(p[1] for p in pos.values()) / len(pos) if pos else 0

    # 3. Gambar Sisi (Edges) dan Gelang (Loops)
    import matplotlib.patches as patches
    import numpy as np

    for u, v in E:
        if u == v: # LOGIKA GELANG (LOOP)
            x, y = pos[u]
            dx, dy = x - center_x, y - center_y
            dist = np.hypot(dx, dy)
            if dist == 0: dx, dy, dist = 0, 1, 1 # Default jika di tengah persis

            # Dorong posisi gelang ke luar
            loop_x = x + (dx / dist) * 0.15
            loop_y = y + (dy / dist) * 0.15

            circle = patches.Circle((loop_x, loop_y), radius=0.15, fill=False, color='midnightblue', linewidth=2, zorder=0)
            ax.add_patch(circle)

        else: # LOGIKA SISI & SISI GANDA (MULTIPLE EDGES)
            pair = tuple(sorted((u, v)))
            total = edge_counts[pair]
            current = drawn_counts[pair]

            if total == 1:
                # Sisi tunggal lurus
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color='darkgray', width=2)
            else:
                # Sisi ganda melengkung
                sign = 1 if current % 2 == 0 else -1
                step = (current + 1) // 2
                rad = 0.2 * sign * step
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
col1, col2 = st.columns([1, 2.5])

# -----------------------------------------------------------
# KOLOM 1: PANEL INPUT DATA
# -----------------------------------------------------------
with col1:
    st.subheader("Input Data Graf")
    st.markdown("---")

    # INPUT 1: Himpunan Simpul
    st.markdown("**Himpunan Simpul/Titik (V):**")
    teks_simpul = st.text_input(
        label="Masukkan Himpunan Simpul/Titik (V):",
        value="",
        placeholder="Contoh: a, b, c, d",
        help="Masukkan nama simpul dipisahkan koma atau spasi. Contoh: a, b, c, d",
        label_visibility="collapsed"
    )
    st.caption("💡 Pisahkan dengan koma atau spasi. Contoh: `a, b, c, d`")

    st.markdown("&nbsp;", unsafe_allow_html=True)  # Spasi vertikal

    # INPUT 2: Himpunan Sisi
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
    tombol_proses = st.button(
        "🔍 Analisis Graf",
        type="primary",
        width='stretch'
    )

    st.markdown("---")

    # BOX PETUNJUK PENGGUNAAN
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
    if teks_simpul.strip():  # Auto-render jika ada input

        # --- PARSING INPUT ---
        simpul_list = parse_simpul(teks_simpul)
        sisi_list = parse_sisi(teks_sisi)

        # Validasi: pastikan semua simpul pada sisi ada di daftar simpul
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
        G = nx.MultiGraph()
        G.add_nodes_from(simpul_list)  # Tambahkan semua simpul (termasuk terisolasi)
        for u, v in sisi_valid:
            G.add_edge(u, v)           # MultiGraph otomatis izinkan duplikat dan loop

        # --- INFO RINGKAS GRAF ---
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
            st.pyplot(fig, use_container_width=False)  # False = tidak stretch ke lebar penuh
        plt.close(fig)  # Tutup figure agar tidak memory leak

        # --- NOTASI FORMAL ---
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

        # Tampilkan derajat setiap simpul
        df_derajat = pd.DataFrame(
            [(v, derajat[v]) for v in simpul_list],
            columns=["Simpul (v)", "Derajat d(v)"]
        )

        # Tampilkan tabel derajat — paksa center via HTML (bypass Streamlit alignment)
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

            # Keterangan setiap sisi
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
