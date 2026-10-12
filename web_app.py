"""
Đớp Tool — Tối ưu hoá kết cấu dàn/khung thép bằng Phép tính Biến phân
Chạy:  streamlit run web_app.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from fem_core import build_docx_bytes
import viz_results as vz
from opt_core import (
    BEAM_BCS, COLUMN_BCS, STEEL_E, STEEL_RHO, STEEL_SIGMA,
    BeamProblem, StructInput, make_family, optimize_beam, optimize_column, optimize_frame,
    ten_bar_truss_example, two_bar_kkt_solution, weight_to_mass_kg,
)
from section_3d import section_3d_comparison, section_3d_note
try:
    from toi_uu_gian_10_thanh_GA_v2 import (
        ThamSoGA,
        khoi_luong as ga_khoi_luong,
        phan_tich_gian as ga_phan_tich_gian,
        thuat_toan_di_truyen,
        toi_uu_gradient,
    )
    GA_IMPORT_ERROR = None
except Exception:
    GA_IMPORT_ERROR = None
    try:
        raise
    except Exception as ex:
        GA_IMPORT_ERROR = ex
    ThamSoGA = None
    ga_khoi_luong = ga_phan_tich_gian = thuat_toan_di_truyen = toi_uu_gradient = None
st.set_page_config(page_title="Đớp Tool", page_icon="🦈", layout="wide")

import opt_core as _oc
if "user_nodal" not in getattr(_oc.StructInput, "__dataclass_fields__", {}):
    st.error("File `opt_core.py` trên máy bạn là phiên bản CŨ (thiếu trường `user_nodal`). "
             "Hãy giải nén lại toàn bộ `dop_tool.zip` và ghi đè TẤT CẢ file (opt_core.py, web_app.py, "
             "legacy_plots.py, viz_results.py, fem_core.py), xoá thư mục `__pycache__`, rồi chạy lại Streamlit.")
    st.stop()

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e67300", "#2ca02c", "#d62728", "#7f7f7f"
KIP_TO_KN = 4.4482216152605
LB_TO_KN = KIP_TO_KN / 1000.0
KN_TO_LB = 1.0 / LB_TO_KN
TEN_BAR_GA_TO_APP_ORDER = np.array([0, 1, 2, 3, 4, 5, 7, 6, 9, 8], dtype=int)


# ══════════════════════════════════════════════════════
#  CSS & tiện ích
# ══════════════════════════════════════════════════════
def inject_css():
    """Theme-aware CSS for Streamlit light/dark themes."""
    try:
        theme_type = getattr(getattr(st.context, "theme", None), "type", "light")
    except Exception:
        theme_type = "light"

    dark = theme_type == "dark"
    if dark:
        paper, ink, muted = "#0e1117", "#f1f5f9", "#aab4c0"
        green, green_soft, rule = "#69c08a", "#173526", "#303844"
        card, input_bg = "#151922", "#11151c"
        good, bad = "#73d69a", "#ff8d86"
    else:
        paper, ink, muted = "#f4f6f3", "#202a25", "#66736c"
        green, green_soft, rule = "#285c45", "#e7efe9", "#cfd8d2"
        card, input_bg = "#ffffff", "#ffffff"
        good, bad = "#326746", "#b64237"

    st.markdown(f"""
    <style>
    #MainMenu, footer {{display:none !important;}}

    :root {{
        --course-paper:{paper}; --course-ink:{ink}; --course-muted:{muted};
        --course-green:{green}; --course-green-soft:{green_soft};
        --course-rule:{rule}; --course-card:{card}; --course-input:{input_bg};
        --course-good:{good}; --course-bad:{bad};
    }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background:var(--course-paper) !important;
        color:var(--course-ink) !important;
    }}
    .block-container {{padding-top:1rem; max-width:1520px;}}

    h1,h2,h3,h4,h5,h6,.stMarkdown,p,label {{
        color:var(--course-ink);
    }}
    h1,h2,h3 {{font-family:Cambria,Georgia,serif; letter-spacing:0;}}
    h1 {{font-size:2.05rem;}}

    .course-kicker {{
        display:flex; align-items:center; gap:.65rem; margin:.25rem 0 .45rem;
        color:var(--course-muted); font-size:.76rem; font-weight:650;
    }}
    .course-code {{
        padding:.24rem .46rem; border-radius:3px;
        background:var(--course-green); color:#fff;
    }}

    .metric-strip {{
        display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:10px; margin:.35rem 0 .9rem;
    }}
    .metric-card {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:10px 12px; background:var(--course-card);
        color:var(--course-ink); box-shadow:none;
    }}
    .metric-label {{color:var(--course-muted); font-size:.74rem; margin-bottom:4px; text-transform:uppercase;}}
    .metric-value {{font-weight:700; font-size:1.12rem; color:var(--course-ink);}}
    .metric-good .metric-value {{color:var(--course-good);}}
    .metric-bad .metric-value {{color:var(--course-bad);}}

    .stTabs [data-baseweb="tab-list"] {{
        gap:.15rem; background:transparent; border-bottom:1px solid var(--course-rule);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:5px 5px 0 0; padding:.62rem .85rem; color:var(--course-muted);
    }}
    .stTabs [data-baseweb="tab"] p {{margin:0; font-weight:600;}}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background:var(--course-green-soft); color:var(--course-green);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{background:var(--course-green);}}

    .report-shell {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:.5rem .65rem; background:var(--course-card);
        color:var(--course-ink);
    }}
    .section-callout {{
        border-left:3px solid var(--course-green);
        background:var(--course-green-soft); padding:.8rem 1rem;
        border-radius:4px; margin:.6rem 0 1rem; color:var(--course-ink);
    }}
    .example-grid {{
        display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:.9rem; margin-top:.8rem;
    }}
    .example-card {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:.9rem 1rem; background:var(--course-card);
        color:var(--course-ink);
    }}
    .example-card h4 {{margin:0 0 .45rem; font-size:1rem; color:var(--course-ink);}}
    .example-card p {{margin:0; color:var(--course-muted); font-size:.9rem;}}
    .example-card .chip {{
        display:inline-block; margin-top:.6rem; background:var(--course-green-soft);
        color:var(--course-green); border-radius:3px; padding:.2rem .6rem;
        font-size:.74rem; font-weight:600;
    }}

    /* Streamlit/BaseWeb controls now follow the same palette. */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background:var(--course-input) !important;
        color:var(--course-ink) !important;
        border-color:var(--course-rule) !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="input"] input {{
        color:var(--course-ink) !important;
    }}
    [role="listbox"], [role="option"] {{
        background:var(--course-card) !important;
        color:var(--course-ink) !important;
    }}
    [role="option"]:hover {{background:var(--course-green-soft) !important;}}

    textarea {{
        font-family:Consolas,"Courier New",monospace !important;
        border-radius:4px !important;
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {{
        color:var(--course-ink) !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color:var(--course-muted) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def metrics(items):
    html = '<div class="metric-strip">'
    for it in items:
        label, value = it[0], it[1]
        cls = it[2] if len(it) > 2 else ""
        html += f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def clean_rows(df, ncols, int_cols=()):
    out = []
    if df is None:
        return out
    for _, row in df.iterrows():
        vals = list(row.values[:ncols])
        if any(pd.isna(v) for v in vals):
            continue
        out.append(tuple(int(v) if k in int_cols else float(v) for k, v in enumerate(vals)))
    return out


def section_figure(fam, area: float, state_label: str = "", view_extent: float | None = None) -> go.Figure:
    dimensions = fam.cross_section_dimensions(area)
    fig = go.Figure()
    try:
        _dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    except Exception:
        _dark = False
    fill = "#263b52" if _dark else "#dcecff"
    edge = "#7db7ff" if _dark else "#1f5f99"
    dim_color = "#b8c4d1" if _dark else "#485766"
    dim_text = "#f1f5f9" if _dark else "#202a35"
    hollow_fill = "#0e1117" if _dark else "#ffffff"

    def dimension_line(x1, y1, x2, y2, text):
        fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           text=text, showarrow=True, arrowhead=3, startarrowhead=3, arrowsize=1,
                           arrowwidth=1.2, arrowcolor="#485766", font=dict(size=11, color=dim_text),
                           bgcolor=("rgba(21,25,34,0.94)" if _dark else "rgba(255,255,255,0.9)"))

    if fam.shape_kind == "rect":
        width, height = dimensions["b"], dimensions["h"]
        fig.add_shape(type="rect", x0=-width / 2, x1=width / 2, y0=-height / 2, y1=height / 2,
                      fillcolor=fill, line=dict(color=edge, width=2))
        dimension_line(-width / 2, -height * 0.62, width / 2, -height * 0.62,
                       f"b = {width * 1000:.1f} mm")
        dimension_line(-width * 0.72, -height / 2, -width * 0.72, height / 2,
                       f"h = {height * 1000:.1f} mm")
        extent = max(width, height)
    elif fam.shape_kind in ("circle", "tube"):
        diameter = dimensions["D"]
        radius = diameter / 2
        fig.add_shape(type="circle", x0=-radius, x1=radius, y0=-radius, y1=radius,
                      fillcolor=fill, line=dict(color=edge, width=2))
        if fam.shape_kind == "tube":
            inner_radius = radius - dimensions["t"]
            fig.add_shape(type="circle", x0=-inner_radius, x1=inner_radius,
                          y0=-inner_radius, y1=inner_radius,
                          fillcolor=hollow_fill, line=dict(color=edge, width=1.5))
        dimension_line(-radius, -radius * 1.35, radius, -radius * 1.35,
                       f"D = {diameter * 1000:.1f} mm")
        extent = diameter
    elif fam.shape_kind == "ibeam":
        height, width, thickness = dimensions["h"], dimensions["b_f"], dimensions["t_f"]
        for center_y in (-height / 2 + thickness / 2, height / 2 - thickness / 2):
            fig.add_shape(type="rect", x0=-width / 2, x1=width / 2,
                          y0=center_y - thickness / 2, y1=center_y + thickness / 2,
                          fillcolor=fill, line=dict(color=edge, width=2))
        fig.add_shape(type="line", x0=0, x1=0, y0=-height / 2 + thickness, y1=height / 2 - thickness,
                      line=dict(color=dim_color, width=1.2, dash="dash"))
        dimension_line(-width / 2, height * 0.68, width / 2, height * 0.68,
                       f"b_f = {width * 1000:.1f} mm")
        dimension_line(-width * 0.68, -height / 2, -width * 0.68, height / 2,
                       f"h = {height * 1000:.1f} mm")
        extent = max(width, height)
    else:
        extent = 1.0
        fig.add_shape(type="rect", x0=-0.35, x1=0.35, y0=-0.35, y1=0.35,
                      fillcolor="rgba(220,236,255,0.35)", line=dict(color="#778899", width=2, dash="dash"))
        fig.add_annotation(x=0, y=0, text="Chưa khai báo hình học", showarrow=False,
                           font=dict(size=12, color=dim_color))

    padding = max(extent, view_extent or 0.0) * 0.85
    title = f"Mặt cắt ngang · A = {area * 1e4:.2f} cm²"
    if state_label:
        title = f"{state_label} · A = {area * 1e4:.2f} cm²"
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center",
                                 font=dict(size=14)),
                      height=280, margin=dict(l=18, r=18, t=42, b=18),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    fig.update_xaxes(range=[-padding, padding], visible=False, fixedrange=True, constrain="domain")
    fig.update_yaxes(range=[-padding, padding], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    return fig


def show_section_preview(fam, area: float, key: str, state_label: str = "",
                         view_extent: float | None = None) -> None:
    dimensions = fam.cross_section_dimensions(area)
    st.plotly_chart(section_figure(fam, area, state_label, view_extent), width="stretch",
                    config={"displayModeBar": False}, key=key)
    if fam.shape_kind == "rect":
        ratio = fam.shape_parameters[0]
        shape_note = f"Chữ nhật đặc: b/h = {ratio:g}; b = {dimensions['b'] * 1000:.1f} mm, h = {dimensions['h'] * 1000:.1f} mm."
    elif fam.shape_kind == "circle":
        shape_note = f"Tròn đặc: D = {dimensions['D'] * 1000:.1f} mm."
    elif fam.shape_kind == "tube":
        shape_note = (f"Ống thành mỏng: D = {dimensions['D'] * 1000:.1f} mm, "
                      f"t = {dimensions['t'] * 1000:.1f} mm; A ≈ πDt.")
    elif fam.shape_kind == "ibeam":
        shape_note = (f"Chữ I lý tưởng hóa: h = {dimensions['h'] * 1000:.1f} mm cố định; "
                      "A là tổng diện tích hai cánh, bụng bỏ qua; bề rộng cánh chỉ minh họa.")
    else:
        shape_note = "Họ tuỳ chỉnh chưa xác định hình học vật lý của tiết diện."
    st.caption(f"{shape_note} Mô hình: I = αAⁿ (n = {fam.n:g}); W = βAⁿ⁻ᵞ (γ = {fam.gamma:g}).")


def _section_extent(fam, area: float) -> float:
    if fam.shape_kind == "custom":
        return 1.0
    return max(fam.cross_section_dimensions(area).values())


def show_section_comparison(fam, area_before: float, area_after: float, key: str) -> None:
    view_extent = max(_section_extent(fam, area_before), _section_extent(fam, area_after))
    before, after = st.columns(2)
    with before:
        show_section_preview(fam, area_before, f"{key}_before", "Trước", view_extent)
    with after:
        show_section_preview(fam, area_after, f"{key}_after", "Sau", view_extent)


def show_section_3d(fam, length: float, area_before, area_after, key: str,
                    orientation: str = "horizontal") -> None:
    st.plotly_chart(section_3d_comparison(fam, length, area_before, area_after, orientation), width="stretch",
                    config={"displayModeBar": False}, key=key)
    axis_note = "Chiều dài cột dựng theo phương đứng." if orientation == "vertical" else \
        "Chiều dài cấu kiện dựng theo phương ngang."
    st.caption(section_3d_note(fam) + f" {axis_note} Màu xám = trước, xanh = sau tối ưu.")


def report_block(text: str, key: str, title: str):
    with st.expander("📄 Thuyết minh tính toán", expanded=False):
        st.text_area("Nội dung", text, height=420, key=f"ta_{key}", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải .txt", text.encode("utf-8"), file_name=f"{key}.txt", key=f"dl_txt_{key}")
        try:
            c2.download_button("⬇️ Tải .docx", build_docx_bytes(text, title), file_name=f"{key}.docx",
                               key=f"dl_docx_{key}")
        except Exception as ex:  # python-docx thiếu
            c2.caption(f"Không tạo được .docx: {ex}")


def family_selector(prefix: str, default="Ống tròn mỏng"):
    opts = ["Chữ nhật đặc", "Tròn đặc", "Ống tròn mỏng", "Chữ I (cánh mỏng, h cố định)", "Tuỳ chỉnh I = α·Aⁿ"]
    st.session_state.setdefault(f"{prefix}_fam", default)
    c1, c2 = st.columns([2, 1.2])
    kind = c1.selectbox("Họ tiết diện", opts, key=f"{prefix}_fam",
                        help="Biến thiết kế duy nhất là diện tích A(x); I(x) và W(x) suy ra từ họ tiết diện (I = α·Aⁿ).")
    if kind == "Chữ nhật đặc":
        st.session_state.setdefault(f"{prefix}_r", 0.5)
        r = c2.number_input("b/h", 0.1, 2.0, step=0.05, key=f"{prefix}_r")
        fam = make_family("rect", ratio=r)
    elif kind == "Tròn đặc":
        fam = make_family("circle")
    elif kind == "Ống tròn mỏng":
        st.session_state.setdefault(f"{prefix}_t", 0.05)
        r = c2.number_input("t/D", 0.01, 0.2, step=0.01, key=f"{prefix}_t")
        fam = make_family("tube", ratio=r)
    elif kind.startswith("Chữ I"):
        st.session_state.setdefault(f"{prefix}_h", 0.4)
        h = c2.number_input("h (m)", 0.05, 3.0, step=0.05, key=f"{prefix}_h")
        fam = make_family("ibeam", h=h)
    else:
        cc = st.columns(4)
        st.session_state.setdefault(f"{prefix}_n", 2.0)
        st.session_state.setdefault(f"{prefix}_al", 0.08)
        st.session_state.setdefault(f"{prefix}_ga", 0.5)
        st.session_state.setdefault(f"{prefix}_be", 0.17)
        n = cc[0].number_input("n", 0.5, 3.0, step=0.1, key=f"{prefix}_n")
        al = cc[1].number_input("α", 1e-4, 10.0, step=0.01, format="%.4f", key=f"{prefix}_al")
        ga = cc[2].number_input("γ", 0.0, 1.5, step=0.05, key=f"{prefix}_ga")
        be = cc[3].number_input("β", 1e-4, 10.0, step=0.01, format="%.4f", key=f"{prefix}_be")
        fam = make_family("custom", n=n, alpha=al, gamma=ga, beta=be)

    with st.expander("Xem hình dạng tiết diện", expanded=True):
        area_cm2 = st.slider("Diện tích minh hoạ A (cm²)", 5, 500, 100, 5, key=f"{prefix}_section_area")
        show_section_preview(fam, area_cm2 * 1e-4, f"{prefix}_section_preview")
    return fam


def fig_layout(fig, title="", h=380, xt="", yt=""):
    fig.update_layout(title=title, height=h, margin=dict(l=50, r=20, t=50, b=40), template="plotly_white",
                      legend=dict(orientation="h", y=1.12, x=0), xaxis_title=xt, yaxis_title=yt)
    return fig


# ══════════════════════════════════════════════════════
#  TRỰC QUAN HOÁ TRƯỚC / SAU TỐI ƯU
#  Nét đứt = phương án trước tối ưu (mốc), nét liền + tô màu = sau tối ưu.
#  Kích thước vẽ ∝ A^g, g = γ của họ tiết diện (γ = 0, vd chữ I cao cố định → vẽ ∝ A).
# ══════════════════════════════════════════════════════
SAVE_FILL, UP_FILL, BODY_FILL = "rgba(44,160,44,.22)", "rgba(230,115,0,.45)", "rgba(31,119,180,.40)"


def _shape_exp(fam) -> float:
    return float(fam.gamma) if fam.gamma > 1e-9 else 1.0


def _band_xy(x, lo, hi):
    """Hai dải đối xứng (+ và −) giữa lo và hi, ngăn bằng None để vẽ trong 1 trace."""
    xs = np.concatenate([x, x[::-1]])
    up = np.concatenate([hi, lo[::-1]])
    dn = np.concatenate([-hi, -lo[::-1]])
    return (list(xs) + [None] + list(xs)), (list(up) + [None] + list(dn))


def fig_before_after_member(x, A_ref, A_opt, fam, ref_label, title, xt="x (m)", h=330):
    x = np.asarray(x, float)
    A_opt = np.asarray(A_opt, float)
    A_ref = np.full_like(x, float(A_ref)) if np.isscalar(A_ref) else np.asarray(A_ref, float)
    g = _shape_exp(fam)
    s = max(float(np.max(A_opt)), float(np.max(A_ref)), 1e-30)
    ho, hr = (A_opt / s) ** g, (A_ref / s) ** g
    fig = go.Figure()
    bx, by = _band_xy(x, np.minimum(ho, hr), hr)
    fig.add_scatter(x=bx, y=by, fill="toself", mode="lines", line=dict(width=0), fillcolor=SAVE_FILL,
                    name="Vật liệu tiết kiệm được", hoverinfo="skip")
    # thân tối ưu: một polygon kín từ −h_o đến +h_o
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(ho) + list(-ho[::-1]), fill="toself", mode="lines",
                    line=dict(color=BLUE, width=2.5), fillcolor=BODY_FILL, name="Sau tối ưu (nét liền)", hoverinfo="skip")
    if np.any(ho > hr + 1e-9):
        bx, by = _band_xy(x, hr, np.maximum(ho, hr))
        fig.add_scatter(x=bx, y=by, fill="toself", mode="lines", line=dict(width=0), fillcolor=UP_FILL,
                        name="Phần phải tăng thêm", hoverinfo="skip")
    for sgn, show in ((1, True), (-1, False)):
        fig.add_scatter(x=x, y=sgn * hr, mode="lines", line=dict(color=vz.before_color(), width=2, dash="dash"),
                        name=ref_label, showlegend=show, hoverinfo="skip")
    fig.add_hline(y=0, line=dict(color=GREY, width=1, dash="dot"))
    fig.update_yaxes(range=[-1.18, 1.18], showticklabels=False, zeroline=False)
    fig.update_layout(legend=dict(orientation="h", y=1.2, x=0))
    return fig_layout(fig, title, h, xt, "")


def _poly_quad(p0, p1, n, h0, h1):
    return [p0 + n * h0, p1 + n * h1, p1 - n * h1, p0 - n * h0, p0 + n * h0]


def plot_structure_before_after(res, title="Tiết diện từng thanh — trước (nét đứt) so với sau tối ưu (nét liền)"):
    inp, model, opt = res.inp, res.model, res.opt
    fam = inp.fam
    g = _shape_exp(fam)
    A_ref = float(opt.A_uniform)
    Amax = max(float(np.max(opt.A)), A_ref)
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    wfull = 0.05 * span                                     # bề rộng vẽ của tiết diện lớn nhất
    half = lambda A: 0.5 * wfull * (float(A) / Amax) ** g

    def add_poly(store, pts):
        store[0] += [float(q[0]) for q in pts] + [None]
        store[1] += [float(q[1]) for q in pts] + [None]

    blue, orange, dash = [[], []], [[], []], [[], []]
    for k, (i, j) in enumerate(inp.members):               # mốc: 1 hình chữ nhật nét đứt cho mỗi thanh
        p0, p1 = np.array(inp.nodes[i], float), np.array(inp.nodes[j], float)
        d = p1 - p0; Lm = float(np.hypot(*d)); n = np.array([-d[1], d[0]]) / Lm
        add_poly(dash, _poly_quad(p0, p1, n, half(A_ref), half(A_ref)))
    for e in range(model.n_el):                            # sau tối ưu: từng phần tử (thấy được vút thon trong thanh)
        p0, p1 = model.xy[model.conn[e, 0]], model.xy[model.conn[e, 1]]
        d = p1 - p0; Lm = float(np.hypot(*d)); n = np.array([-d[1], d[0]]) / Lm
        h = half(opt.A[e])
        add_poly(orange if opt.A[e] > A_ref * 1.0005 else blue, _poly_quad(p0, p1, n, h, h))

    fig = go.Figure()
    fig.add_scatter(x=blue[0], y=blue[1], fill="toself", mode="lines", fillcolor=BODY_FILL,
                    line=dict(color=BLUE, width=1.6), name="Sau tối ưu — nhỏ hơn mốc", hoverinfo="skip")
    fig.add_scatter(x=orange[0], y=orange[1], fill="toself", mode="lines", fillcolor=UP_FILL,
                    line=dict(color=ORANGE, width=1.6), name="Sau tối ưu — lớn hơn mốc", hoverinfo="skip")
    fig.add_scatter(x=dash[0], y=dash[1], mode="lines", line=dict(color=vz.before_color(), width=1.6, dash="dash"),
                    name="Trước tối ưu (tiết diện đều)", hoverinfo="skip")
    # điểm hover tại giữa mỗi thanh
    mx, my, tx = [], [], []
    for k, (i, j) in enumerate(inp.members):
        sel = res.member_of_el == k
        mx.append((inp.nodes[i][0] + inp.nodes[j][0]) / 2); my.append((inp.nodes[i][1] + inp.nodes[j][1]) / 2)
        a_o = float(res.A_member[k])
        tx.append(f"Thanh {k}: trước {A_ref * 1e4:.2f} cm² → sau {a_o * 1e4:.2f} cm² ({(a_o / A_ref - 1) * 100:+.0f}%)")
    fig.add_scatter(x=mx, y=my, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"), text=tx,
                    hoverinfo="text", showlegend=False)
    _add_nodes_supports_loads(fig, inp, span, xs, ys)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    fig = fig_layout(fig, title, 580)
    fig.update_layout(margin=dict(l=50, r=20, t=105, b=40), legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
    return fig


# ══════════════════════════════════════════════════════
#  TAB 0 — LÝ THUYẾT
# ══════════════════════════════════════════════════════
@st.cache_resource
def _cached_lecture_examples():
    return _oc.lecture_examples()


def _lecture_example_figure(key: str, item: dict):
    result = item["result"]
    if key == "two_bar":
        B = item["geometry"]["B"]
        h = result["h"]
        nodes = [(0.0, 0.0), (2 * B, 0.0), (B, h)]
        fig = go.Figure()
        for member_id, (i, j) in enumerate(((0, 2), (1, 2)), start=1):
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj], mode="lines",
                line=dict(color="#2676a8", width=5), showlegend=False,
                hovertemplate=f"Thanh {member_id}<br>N = −{result['axial_force']:.2f} kN (nén)<extra></extra>",
            ))
            fig.add_annotation(x=(xi + xj) / 2, y=(yi + yj) / 2,
                               text=f"Thanh {member_id}: N = −{result['axial_force']:.1f} kN",
                               showarrow=False, bgcolor="rgba(255,255,255,.88)", font=dict(size=11))
        fig.add_trace(go.Scatter(x=[point[0] for point in nodes], y=[point[1] for point in nodes],
                                 mode="markers", marker=dict(size=10, color="#263746"), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[0.0, 2 * B], y=[-0.06 * h, -0.06 * h], mode="markers",
                                 marker=dict(symbol="triangle-up", size=15, color="#c44536"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=B, y=h, ax=B, ay=h + 0.45 * h, showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor="#d94841",
                           text=f"P = {item['geometry']['P_kN']:.0f} kN")
        fig.add_annotation(x=B, y=-0.17 * h, text=f"2B = {2 * B:.2f} m", showarrow=False,
                           font=dict(size=11, color="#4b5d70"))
        fig.update_layout(title="Sơ đồ hình học và cân bằng nút đỉnh", height=350,
                          margin=dict(l=20, r=20, t=55, b=25), template="plotly_white")
        fig.update_xaxes(title="x (m)", range=[-0.25 * B, 2.25 * B], zeroline=False)
        fig.update_yaxes(title="y (m)", range=[-0.35 * h, 1.75 * h], scaleanchor="x", scaleratio=1,
                         zeroline=False)
        return fig, "Đối xứng nên hai thanh có cùng lực nén; tăng h làm thanh đứng hơn, giảm lực dọc nhưng tăng chiều dài/vật liệu."

    if key == "column":
        fig = go.Figure()
        fig.add_scatter(x=result.x, y=result.A_opt * 1e4, mode="lines", name="A*(x) — tối ưu",
                        line=dict(color=BLUE, width=3),
                        hovertemplate="x=%{x:.2f} m<br>A=%{y:.2f} cm²<extra></extra>")
        fig.add_scatter(x=[0, result.L], y=[result.A_uniform * 1e4] * 2, mode="lines",
                        name="Cột đều — cùng V", line=dict(color=ORANGE, width=2, dash="dash"),
                        hovertemplate="A đều=%{y:.2f} cm²<extra></extra>")
        fig.update_layout(title="Cột khớp–khớp: dồn vật liệu theo dạng mất ổn định", height=350,
                          margin=dict(l=55, r=20, t=55, b=45), template="plotly_white",
                          legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))
        fig.update_xaxes(title="x dọc cột (m)")
        fig.update_yaxes(title="Diện tích A (cm²)", rangemode="tozero")
        return fig, "V cố định: cột tối ưu tăng A ở vùng mode uốn lớn và giảm A gần gối; đường gạch là cột đều cùng thể tích."

    if key == "beam":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                            subplot_titles=("Diện tích dọc dầm", "Đường võng (hướng xuống dương)"))
        fig.add_trace(go.Scatter(x=result.x_mid, y=result.A_opt * 1e4, mode="lines",
                                 name="A*(x) — tối ưu", line=dict(color=BLUE, width=3),
                                 hovertemplate="x=%{x:.2f} m<br>A=%{y:.2f} cm²<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, result.prob.L], y=[result.A_uniform * 1e4] * 2, mode="lines",
                                 name="Dầm đều — cùng giới hạn", line=dict(color=ORANGE, width=2, dash="dash"),
                                 hovertemplate="A đều=%{y:.2f} cm²<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.xs, y=result.deflection_opt * 1e3, mode="lines",
                                 name="Võng tối ưu", line=dict(color=RED, width=2.5),
                                 hovertemplate="x=%{x:.2f} m<br>δ=%{y:.2f} mm<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.xs, y=result.deflection_uni * 1e3, mode="lines",
                                 name="Võng dầm đều", line=dict(color="#697586", width=2, dash="dash"),
                                 hovertemplate="x=%{x:.2f} m<br>δ=%{y:.2f} mm<extra></extra>"), row=2, col=1)
        if result.prob.delta_max is not None:
            limit_mm = result.prob.delta_max * 1e3
            fig.add_hline(y=limit_mm, line=dict(color="#4b5563", dash="dot"), row=2, col=1)
            fig.add_hline(y=-limit_mm, line=dict(color="#4b5563", dash="dot"), row=2, col=1)
        fig.update_layout(title="Dầm tựa đơn: vật liệu theo mômen ảo và đáp ứng võng", height=540,
                          margin=dict(l=55, r=20, t=75, b=75), template="plotly_white",
                          legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"))
        fig.update_xaxes(title_text="x dọc dầm (m)", row=2, col=1)
        fig.update_yaxes(title_text="A (cm²)", rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="Độ võng (mm)", autorange="reversed", row=2, col=1)
        return fig, "Vật liệu được ưu tiên nơi |M·m̄| lớn; đường chấm trong đồ thị võng là ±δ giới hạn."

    raise ValueError(f"Không có hình minh họa cho ví dụ: {key}")


def tab_theory():
    st.header("Cơ sở toán học")
    st.markdown("""
**Ý tưởng.** Khi biến thiết kế là *hàm liên tục* $A(x)$ (dầm vút thon, cột biến tiết diện) ta không tìm một vài con số mà tìm *cả một hàm*.
Đó là bài toán cực trị phiếm hàm. Đớp Tool dùng phép tính biến phân để ra **dạng đóng** của $A^*(x)$, rồi dùng FEM + tiêu chuẩn tối ưu để
(i) kiểm chứng và (ii) mở rộng cho các sơ đồ siêu tĩnh, khung và dàn, nơi không còn nghiệm đóng.
""")
    st.subheader("1. Biến phân bậc nhất và phương trình Euler–Lagrange")
    st.latex(r"J[y]=\int_{0}^{L}F(x,y,y')\,\mathrm dx\ \to\ \text{stationary}\quad\Rightarrow\quad F_y-\frac{\mathrm d}{\mathrm dx}\big(F_{y'}\big)=0")
    st.markdown(r"""
Chọn hàm thử $y_\varepsilon=y+\varepsilon\eta$, trong đó $\eta$ là biến phân khả dĩ. Với hai đầu cố định, $\eta(a)=\eta(b)=0$.
""")
    st.latex(r"\delta J=\left.\frac{\mathrm dJ[y+\varepsilon\eta]}{\mathrm d\varepsilon}\right|_{\varepsilon=0}=\int_a^b(F_y\eta+F_{y'}\eta')\,\mathrm dx")
    st.latex(r"\delta J=\int_a^b\left(F_y-\frac{\mathrm d}{\mathrm dx}F_{y'}\right)\eta\,\mathrm dx+\left[F_{y'}\eta\right]_a^b")
    st.markdown(r"""
Tích phân từng phần đưa đạo hàm khỏi $\eta$. Theo **bổ đề cơ bản của phép tính biến phân**, nếu $\delta J=0$ với mọi $\eta$ khả dĩ thì
$F_y-\frac{\mathrm d}{\mathrm dx}F_{y'}=0$. Ở đầu tự do, $\eta$ không bị ấn định nên số hạng biên cho điều kiện tự nhiên $F_{y'}=0$;
điều kiện biên cơ bản thì được áp đặt trước lên lớp hàm.
""")
    st.subheader("1.1. Đưa ràng buộc vào bài toán biến phân")
    st.markdown(r"Với ràng buộc toàn cục $G[y]=\int_a^b g(x,y,y')\,\mathrm dx-G_0=0$, lập phiếm hàm Lagrange $H=F+\lambda g$ rồi áp dụng Euler–Lagrange cho $H$. Khi $F$ không chứa $y'$, điều kiện dừng suy biến thành phương trình đại số.")
    st.latex(r"H_y-\frac{\mathrm d}{\mathrm dx}H_{y'}=0,\qquad A_{\min}\leq A(x)\leq A_{\max}")
    st.markdown("Với cận cục bộ, chia miền thành vùng tự do và vùng chạm cận; tại nghiệm phải kiểm tra khả thi, dấu nhân tử và bù trừ, không chỉ phương trình dừng.")
    with st.expander("Ví dụ trên lớp: tối ưu giàn hai thanh và kiểm tra KKT", expanded=False):
        c1, c2, c3 = st.columns(3)
        load_kN = c1.number_input("P (kN)", min_value=0.1, value=150.0, step=10.0, key="tb_P")
        half_span_mm = c2.number_input("B (mm)", min_value=1.0, value=1000.0, step=100.0, key="tb_B")
        stress_MPa = c3.number_input("[σ] (MPa)", min_value=0.1, value=180.0, step=10.0, key="tb_sigma")
        density = st.number_input("Khối lượng riêng ρ (kg/m³)", min_value=1.0, value=7850.0, step=100.0, key="tb_rho")
        example = two_bar_kkt_solution(load_kN, half_span_mm / 1000, stress_MPa * 1000, density)
        metrics([
            ("Chiều cao tối ưu h*", f"{example['h'] * 1000:.1f} mm"),
            ("Góc thanh α*", "45°"),
            ("Diện tích mỗi thanh A*", f"{example['area'] * 1e6:.1f} mm²"),
            ("Khối lượng giàn m*", f"{example['mass']:.2f} kg"),
        ])
        report_block(example["report"], "gian_hai_thanh_kkt", "Thuyết minh — Giàn hai thanh và KKT")
    lecture_examples = _cached_lecture_examples()
    st.markdown('<div class="section-callout"><strong>Bộ ví dụ về áp dụng lý thuyết chương 1-4.</strong> Mỗi ví dụ minh họa một nội dung đã học: giàn hai thanh, ổn định cột và tối ưu dầm vút thon.</div>', unsafe_allow_html=True)
    example_cols = st.columns(len(lecture_examples))
    for idx, (key, item) in enumerate(lecture_examples.items()):
        with example_cols[idx]:
            st.markdown(f"<div class='example-card'><h4>{item['title']}</h4><p>{item['summary']}</p><span class='chip'>{key}</span></div>", unsafe_allow_html=True)
            if st.button(f"Xem {item['title']}", key=f"see_{key}"):
                st.session_state[f"example_report_{key}"] = item["report"]
    for key, item in lecture_examples.items():
        if st.session_state.get(f"example_report_{key}"):
            figure, geometric_note = _lecture_example_figure(key, item)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False},
                            key=f"lecture_figure_{key}")
            st.caption(geometric_note)
            report_block(st.session_state[f"example_report_{key}"], f"lecture_{key}", item["title"])

    st.subheader("2. Dầm vút thon — cực tiểu thể tích với ràng buộc độ võng")
    st.markdown(r"Họ tiết diện: $I=\alpha A^{n}$ (chữ nhật/tròn/ống đồng dạng $n=2$; chữ I cánh mỏng với $h$ cố định $n=1$). "
                r"Độ võng tại điểm khống chế $x_p$ theo nguyên lý công ảo:")
    st.latex(r"\delta(x_p)=\int_0^L\frac{M(x)\,\bar m(x)}{E\,\alpha\,A^{n}(x)}\,\mathrm dx=\delta_{\max},\qquad V=\int_0^L A\,\mathrm dx\to\min")
    st.latex(r"\tilde F=A+\lambda\,\frac{M\bar m}{E\alpha A^{n}}\ \ (\text{independent of }A')\ \Rightarrow\ 1-\frac{\lambda n M\bar m}{E\alpha A^{n+1}}=0")
    st.latex(r"\boxed{A^{*}(x)=\Big[\tfrac{\lambda n}{E\alpha}\,M(x)\,\bar m(x)\Big]^{\frac1{n+1}}}")
    st.markdown(r"""
- Tải tập trung tại điểm khống chế: $\bar m\propto M\Rightarrow A^*\propto|M|^{\frac{2}{n+1}}$ (chữ nhật/tròn: $|M|^{2/3}$; chữ I cánh mỏng: $|M|$).
- Kết hợp điều kiện bền $A\ge A_{\mathrm{strength}}(x)=A_W(|M|/[\sigma])$ và $A\ge A_{\min}$: nghiệm KKT là $A=\max(A_{\mathrm{strength}},A_{\min},c\,g^{1/(n+1)})$, $g=\max(M\bar m,0)$; hằng số $c$ giải từ $\delta(c)=\delta_{\max}$.
- Dầm siêu tĩnh: $M$ phụ thuộc $A$ nên không có dạng đóng → dùng FEM + OC (mục 4).
""")
    st.subheader("3. Cột — cực đại tải tới hạn với thể tích cho trước (Lagrange – Keller – Tadjbakhsh)")
    st.latex(r"P_{cr}=\min_w\frac{\int E\alpha A^{n}w''^2\,\mathrm dx}{\int w'^2\,\mathrm dx},\quad \int A\,\mathrm dx=V")
    st.latex(r"\delta_A:\ \ n\,E\alpha A^{n-1}w''^{2}=\lambda\ \ \ (\text{algebraic}),\qquad M=-Pw\ \Rightarrow\ A\propto|w|^{\frac2{n+1}}")
    st.latex(r"w''=-\kappa\,|w|^{\frac{1-n}{n+1}}\,\mathrm{sgn}\,w\ \xrightarrow{\ u=A/A_{\max}\ }\ \xi=I_u\!\Big(\tfrac{n+1}2,\tfrac12\Big)\ \Rightarrow\ u(\xi)=I^{-1}_{\xi}\!\Big(\tfrac{n+1}2,\tfrac12\Big)")
    st.latex(r"\frac{\bar A}{A_{\max}}=\frac{n+1}{n+2},\qquad P_{cr}^*=\frac{(n+1)\,B^{2}\!\big(\tfrac{n+1}2,\tfrac12\big)\,E\alpha\,A_{\max}^{\,n}}{L_{ef}^{2}}")
    st.markdown(r"Với $n=2$, hai đầu khớp: $P^*_{cr}/P_{cr,\mathrm{uniform}}=4/3$ — **kết quả cổ điển của Keller (1960)**, được dùng làm test kiểm chứng. "
                r"Nghiệm E–L cho $A\to0$ tại điểm mômen bằng 0; thực tế phải áp cận dưới $A_{\min}$.")
    st.subheader("4. Phần số: tiêu chuẩn tối ưu (OC) và KKT trên lưới FEM")
    st.latex(r"\min_{A}\ W=\rho\sum A_eL_e\quad\text{s.t.}\quad \frac{|N_e|}{A_e\sigma_N}+\frac{|M_e|}{W_e[\sigma]}\le1,\ \ |\delta_k|\le\delta_k^{\lim}")
    st.latex(r"\nabla_A\mathcal L=\nabla_A W+\sum_j\mu_j\nabla_A g_j+\sum_k\lambda_k\nabla_A h_k=\mathbf 0,\quad g_j\leq0,\quad\mu_j\geq0,\quad\mu_jg_j=0")
    st.latex(r"\frac{\partial\delta_k}{\partial A_e}=-\,\mathbf u_{unit}^{T}\frac{\partial\mathbf K}{\partial A_e}\mathbf u")
    st.markdown("""
Điều kiện KKT gồm: (1) dừng; (2) khả thi nguyên thủy; (3) khả thi đối ngẫu; (4) bù trừ. KKT là điều kiện cần khi thỏa điều kiện chính quy; chỉ kết luận cực tiểu toàn cục khi có thêm giả thiết thích hợp, chẳng hạn tính lồi.

Trong ứng dụng: FEM phân tích; OC/FSD tạo nghiệm; SLSQP tinh chỉnh các ràng buộc chuyển vị với độ nhạy liên hợp. Báo cáo kiểm tra khả thi và một chẩn đoán dừng KKT xấp xỉ trên phần tử tự do; chẩn đoán đó **không thay thế kiểm tra đầy đủ mọi nhân tử và điều kiện bù trừ**. Đối chiếu khung với `fem_core.py` là kiểm tra chéo FEM, không phải kiểm tra KKT.
""")
    st.info("**Giới hạn:** Euler–Bernoulli, đàn hồi tuyến tính, bỏ qua biến dạng cắt, ổn định cục bộ, P-Δ, mỏi, tải bản thân và chi tiết liên kết. "
            "Tiết diện thực phải làm tròn theo thép hình rồi kiểm tra lại theo TCVN/AISC. Nghiệm với kết cấu siêu tĩnh là cực trị địa phương.")


# ══════════════════════════════════════════════════════
#  TAB 1 — CỘT
# ══════════════════════════════════════════════════════
def tab_column():
    st.header("CHƯƠNG 06 · Cột chịu nén biến tiết diện")
    left, right = st.columns([1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ liên kết", list(COLUMN_BCS), format_func=COLUMN_BCS.get, key="c_bc")
        L = st.number_input("Chiều dài L (m)", 0.5, 100.0, 5.0, 0.5, key="c_L")
        E = st.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="c_E") * 1e6
        fam = family_selector("c")
        mode = st.radio("Bài toán", ["Cho thể tích → cực đại P_cr", "Cho P yêu cầu → cực tiểu thể tích"], key="c_mode")
        if mode.startswith("Cho thể"):
            A0 = st.number_input("Diện tích trung bình A₀ = V/L (cm²)", 1.0, 5000.0, 30.0, 1.0, key="c_A0")
            P_req = None
        else:
            P_req = st.number_input("P yêu cầu (kN)", 1.0, 1e7, 500.0, 10.0, key="c_P")
        floor = st.slider("Cận dưới A_min (% của A_max)", 0.0, 20.0, 1.0, 0.5, key="c_floor") / 100.0
        run_num = st.checkbox("Chạy thêm tối ưu số OC (kiểm chứng nghiệm đóng)", True, key="c_num")
        go_btn = st.button("🚀 Tối ưu cột", type="primary", key="c_go")
    if go_btn:
        try:
            with st.spinner("Đang tính..."):
                if P_req is None:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, V=A0 * 1e-4 * L, floor_frac=floor,
                                                                run_numeric=run_num)
                else:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, P_req=P_req, floor_frac=floor,
                                                                run_numeric=run_num)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("c_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu cột**.")
            return
        metrics([
            ("P_cr tối ưu (FEM)", f"{r.P_fem_opt:,.1f} kN"),
            ("P_cr cột đều", f"{r.P_uniform:,.1f} kN"),
            ("Hệ số tăng tải (FEM)", f"×{r.ratio_fem:.3f}", "metric-good"),
            ("Hệ số tăng tải (đóng, không cận)", f"×{r.ratio_closed:.3f}"),
            ("V = ∫A dx", f"{r.V * 1e6:,.0f} cm²·m→ {r.V:.5f} m³"),
            ("A_max", f"{np.max(r.A_opt) * 1e4:.2f} cm²"),
        ])
        if getattr(r, "bimodal", False):
            st.info("**Cột hai đầu ngàm:** nghiệm đóng đơn-mode (A = 0 tại L/4 và 3L/4) biến cột thành cơ cấu nên tải FEM thực "
                    "thấp hơn giá trị đóng (thậm chí thấp hơn cột đều). Profile hiển thị là nghiệm **bimodal** (λ₁ = λ₂) "
                    "giải bằng SLSQP; hệ số \"đóng\" chỉ là cận tham khảo.")
        fig = go.Figure()
        fig.add_scatter(x=r.x / r.L, y=r.A_opt * 1e4, name=("Bimodal SLSQP (FEM kiểm)" if getattr(r, "bimodal", False) else "E–L nghiệm đóng (FEM kiểm)"), line=dict(color=BLUE, width=3))
        if r.A_numeric is not None:
            fig.add_scatter(x=r.x_numeric / r.L, y=r.A_numeric * 1e4, name="OC số", mode="lines",
                            line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, 1], y=[r.A_uniform * 1e4] * 2, name="Cột đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x/L", "A (cm²)"), width="stretch")
        sec_pos = st.columns([0.8, 1.2])[0]
        with sec_pos:
            section_x_ratio = st.slider("Vị trí mặt cắt x/L", 0.0, 1.0, 0.5, 0.01, key="c_section_x")
            section_area = float(np.interp(section_x_ratio, r.x / r.L, r.A_opt))
            st.metric("A tại vị trí chọn", f"{section_area * 1e4:.2f} cm²")
        ref_mode = st.radio("Mốc so sánh (nét đứt)", ["Cột đều cùng P_cr (vật liệu cần để chịu cùng tải)",
                                                     "Cột đều cùng thể tích V"], horizontal=True, key="c_ref")
        if ref_mode.startswith("Cột đều cùng P_cr"):
            A_ref = r.A_uniform * (r.P_fem_opt / r.P_uniform) ** (1.0 / r.fam.n)   # P_cr ∝ A^n
            ref_lab = "Trước tối ưu: cột đều, cùng P_cr"
            note = (f"Cột đều chịu cùng P_cr = {r.P_fem_opt:,.0f} kN cần V = {A_ref * r.L * 1e6:,.0f} cm²·m; "
                    f"cột tối ưu chỉ cần {r.V * 1e6:,.0f} cm²·m → **tiết kiệm {(1 - r.V / (A_ref * r.L)) * 100:.1f}% vật liệu**.")
        else:
            A_ref = r.A_uniform
            ref_lab = "Trước tối ưu: cột đều, cùng thể tích"
            note = (f"Cùng thể tích, cột tối ưu chịu P_cr = {r.P_fem_opt:,.0f} kN so với {r.P_uniform:,.0f} kN của cột đều "
                    f"(**×{r.ratio_fem:.3f}**).")
        show_section_comparison(r.fam, A_ref, section_area, "c_section_comparison")
        show_section_3d(r.fam, r.L, A_ref, r.A_opt, "c_section_3d", orientation="vertical")
        st.plotly_chart(fig_before_after_member(r.x, A_ref, r.A_opt, r.fam, ref_lab,
                                                "Hình dạng cột — trước (nét đứt) so với sau tối ưu (nét liền)", "x (m)", 340),
                        width="stretch")
        st.caption(note + " Vùng **xanh lá** = vật liệu bỏ được; vùng **cam** = chỗ phải dày hơn mốc (vật liệu được dồn về "
                          "nơi mômen lớn, nên cột tối ưu luôn phình ra ngoài mốc ở giữa nhịp). "
                          f"Bề rộng vẽ ∝ A^{_shape_exp(r.fam):g}.")
        if r.history:
            cv = go.Figure()
            cv.add_scatter(y=np.array(r.history) / r.P_closed, name="P_cr(OC)/P_closed", line=dict(color=ORANGE))
            cv.add_hline(y=1.0, line_dash="dot")
            st.plotly_chart(fig_layout(cv, "Hội tụ OC so với nghiệm đóng", 300, "vòng lặp", "P/P*"), width="stretch")
        report_block(r.report, "cot_toi_uu", "Thuyết minh — Tối ưu cột")


# ══════════════════════════════════════════════════════
#  TAB 2 — DẦM
# ══════════════════════════════════════════════════════
def tab_beam():
    st.header("CHƯƠNG 07 · Dầm vút thon A(x)")
    left, right = st.columns([1.1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ", list(BEAM_BCS), format_func=BEAM_BCS.get, key="b_bc")
        L = st.number_input("Chiều dài L (m)", 1.0, 100.0, 8.0, 0.5, key="b_L")
        c1, c2 = st.columns(2)
        E = c1.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="b_E") * 1e6
        sig = c2.number_input("[σ] (MPa)", 10.0, 1000.0, STEEL_SIGMA / 1e3, 10.0, key="b_sig") * 1e3
        fam = family_selector("b", default="Chữ I (cánh mỏng, h cố định)")
        st.caption("Tải trọng (hướng xuống > 0)")
        pl_df = st.data_editor(pd.DataFrame({"P (kN)": [100.0], "x (m)": [L / 2]}), num_rows="dynamic",
                               key="b_pl", hide_index=True, width="stretch")
        udl_df = st.data_editor(pd.DataFrame({"q (kN/m)": [10.0], "x1 (m)": [0.0], "x2 (m)": [L]}), num_rows="dynamic",
                                key="b_udl", hide_index=True, width="stretch")
        use_d = st.checkbox("Ràng buộc độ võng", True, key="b_usd")
        d1, d2 = st.columns(2)
        dmax = d1.number_input("|δ| max (mm)", 0.1, 1000.0, round(L * 1000 / 400, 1), 1.0, key="b_dmax") / 1e3
        xc = d2.number_input("Vị trí khống chế x (m)", 0.0, L, L / 2 if bc != "cantilever" else L, 0.5, key="b_xc")
        amin = st.number_input("A_min (cm²)", 0.01, 1000.0, 5.0, 0.5, key="b_amin") * 1e-4
        nel = st.slider("Số phần tử lưới", 40, 200, 100, 10, key="b_nel")
        go_btn = st.button("🚀 Tối ưu dầm", type="primary", key="b_go")
    if go_btn:
        try:
            pls = clean_rows(pl_df, 2)
            udls = clean_rows(udl_df, 3)
            for P, x in pls:
                if not 0 <= x <= L:
                    raise ValueError(f"Tải tập trung tại x={x} nằm ngoài [0, {L}].")
            for q, a, b in udls:
                if not (0 <= a < b <= L):
                    raise ValueError(f"UDL cần 0 ≤ x1 < x2 ≤ {L}.")
            if not pls and not udls:
                raise ValueError("Chưa có tải trọng.")
            prob = BeamProblem(L=L, bc=bc, udls=udls, point_loads=pls, fam=fam, E=E, sigma_a=sig,
                               delta_max=dmax if use_d else None, x_ctrl=xc, A_min=amin, n_elem=nel)
            with st.spinner("Đang tối ưu (FEM + KKT)..."):
                st.session_state["b_res"] = optimize_beam(prob)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("b_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu dầm**.")
            return
        sav = (1 - r.vol_opt / r.vol_uniform) * 100
        items = [("Thể tích tối ưu", f"{r.vol_opt:.4f} m³"), ("Thể tích dầm đều", f"{r.vol_uniform:.4f} m³"),
                 ("Tiết kiệm vật liệu", f"{sav:.1f}%", "metric-good")]
        if r.vol_closed is not None:
            items.append(("Nghiệm đóng E–L", f"{r.vol_closed:.4f} m³"))
        if r.delta_ctrl_opt is not None:
            items.append(("δ tại điểm khống chế", f"{r.delta_ctrl_opt * 1e3:.2f} / {r.prob.delta_max * 1e3:.2f} mm"))
        items.append(("Ràng buộc", "Thoả" if r.opt.feasible else "KHÔNG thoả", "metric-good" if r.opt.feasible else "metric-bad"))
        metrics(items)
        fig = go.Figure()
        xs_step = np.repeat(r.xs, 2)[1:-1]
        fig.add_scatter(x=xs_step, y=np.repeat(r.A_opt, 2) * 1e4, name="FEM + KKT", line=dict(color=BLUE, width=3))
        if r.closed is not None:
            fig.add_scatter(x=r.closed["x"], y=r.closed["A"] * 1e4, name="Nghiệm đóng E–L", line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, r.prob.L], y=[r.A_uniform * 1e4] * 2, name="Dầm đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x (m)", "A (cm²)"), width="stretch")
        sec_pos = st.columns([0.8, 1.2])[0]
        with sec_pos:
            section_x = st.slider("Vị trí mặt cắt x (m)", 0.0, r.prob.L, r.prob.L / 2,
                                  max(r.prob.L / 100, 0.01), key="b_section_x")
            section_area = float(np.interp(section_x, r.x_mid, r.A_opt))
            st.metric("A tại vị trí chọn", f"{section_area * 1e4:.2f} cm²")
        show_section_comparison(r.prob.fam, r.A_uniform, section_area, "b_section_comparison")
        show_section_3d(r.prob.fam, r.prob.L, r.A_uniform, r.A_opt, "b_section_3d")
        xs_step = np.repeat(r.xs, 2)[1:-1]
        st.plotly_chart(fig_before_after_member(xs_step, r.A_uniform, np.repeat(r.A_opt, 2), r.prob.fam,
                                                "Trước tối ưu: dầm đều (cùng ràng buộc)",
                                                "Hình dạng dầm — trước (nét đứt) so với sau tối ưu (nét liền)", "x (m)", 340),
                        width="stretch")
        st.caption(f"Dầm đều thoả cùng ràng buộc cần A = {r.A_uniform * 1e4:.2f} cm² (V = {r.vol_uniform:.4f} m³); "
                   f"dầm tối ưu V = {r.vol_opt:.4f} m³ → **tiết kiệm {sav:.1f}%**. Xanh lá = vật liệu bỏ được, "
                   f"cam = chỗ dày hơn mốc. Chiều cao vẽ ∝ A^{_shape_exp(r.prob.fam):g}.")
        st.subheader("Sơ đồ tải trọng · lực cắt · mômen · chuyển vị")
        fld = vz.beam_fields(r)
        rc = vz.beam_reactions(r, fld)
        if rc:
            metrics([(k, v) for k, v in rc])
        c1, c2 = st.columns(2)
        c1.plotly_chart(vz.plot_beam_loads(r.prob), width="stretch")
        c2.plotly_chart(vz.plot_beam_sfd(fld), width="stretch")
        c3, c4 = st.columns(2)
        c3.plotly_chart(vz.plot_beam_bmd(fld), width="stretch")
        c4.plotly_chart(vz.plot_beam_elastic(r.prob, fld), width="stretch")
        st.caption("Nét đứt trắng = dầm đều (trước tối ưu, cùng ràng buộc); nét liền = dầm vút thon sau tối ưu. "
                   "Quy ước: tải xuống > 0; M > 0 căng thớ dưới (vẽ về phía dưới); V > 0 khi phần bên trái đẩy lên; "
                   "độ võng hướng xuống > 0. Với dầm tĩnh định, V và M của hai phương án trùng nhau (chỉ độ võng khác).")
        report_block(r.report, "dam_toi_uu", "Thuyết minh — Tối ưu dầm vút thon")


# ══════════════════════════════════════════════════════
#  TAB 3 — KHUNG / DÀN
# ══════════════════════════════════════════════════════
@st.cache_resource
def _canvas_component():
    return components.declare_component("dop_canvas", path=os.path.join(HERE, "canvas"))


def _sup(node, kind):
    t = {"pin": (True, True, False), "rollx": (False, True, False), "fixed": (True, True, True)}[kind]
    return dict(node=node, ux=t[0], uy=t[1], rz=t[2])


def _truss_warren():
    nodes = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (1.5, 2.5), (4.5, 2.5), (7.5, 2.5), (10.5, 2.5)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
           (0, 5), (5, 1), (1, 6), (6, 2), (2, 7), (7, 3), (3, 8), (8, 4)]
    return dict(mode="Dàn (nút khớp)", nodes=nodes, members=mem, supports=[_sup(0, "pin"), _sup(4, "rollx")],
                nodal=[(5, 0, -50), (6, 0, -50), (7, 0, -50), (8, 0, -50)], udl=[], limits=[(2, "uy", 30.0)], seg=1)


def _truss_ten_bar():
    inp = ten_bar_truss_example()
    return dict(
        mode="Dàn (nút khớp)",
        nodes=inp.nodes,
        members=inp.members,
        supports=[_sup(node, "pin") for node, _, _, _ in inp.supports],
        nodal=[(node, fx, fy) for node, fx, fy, _ in inp.loads],
        udl=[],
        limits=[(node, direction, limit * 1e3) for node, direction, limit in inp.disp_limits],
        seg=1,
        material=dict(
            E=inp.E / 1e6,
            sigma=inp.sigma_a / 1e3,
            rho=inp.rho,
            amin=inp.A_min * 1e4,
            amax=inp.A_max * 1e4,
            buck=False,
            k=inp.k_eff,
            fs=inp.FS_buckling,
            family="Ống tròn mỏng",
            tube_ratio=0.05,
        ),
        note="Benchmark slide: nhịp 360 in, tải 100 kip tại hai nút đáy; giới hạn chuyển vị 2 in.",
    )


def _frame_portal():
    nodes = [(0, 0), (0, 4), (3, 5), (6, 4), (6, 0)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(4, "fixed")], nodal=[(1, 15, 0)],
                udl=[(1, 20.0), (2, 20.0)], limits=[(2, "uy", 20.0)], seg=4)


def _frame_two_story():
    nodes = [(0, 0), (6, 0), (0, 3.5), (6, 3.5), (0, 7), (6, 7)]
    mem = [(0, 2), (2, 4), (1, 3), (3, 5), (2, 3), (4, 5)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(1, "fixed")], nodal=[(2, 20, 0), (4, 40, 0)],
                udl=[(4, 30.0), (5, 30.0)], limits=[(4, "ux", 35.0)], seg=4)


PRESETS = {"Dàn Warren 12 m": _truss_warren, "Dàn 10 thanh chuẩn (slide)": _truss_ten_bar,
           "Khung cổng 1 nhịp (mái dốc)": _frame_portal, "Khung 2 tầng 1 nhịp": _frame_two_story}


def _load_preset(name):
    p = PRESETS[name]()
    st.session_state["geo"] = dict(nodes=[dict(x=float(x), y=float(y)) for x, y in p["nodes"]],
                                   elements=[dict(i=i, j=j) for i, j in p["members"]], supports=p["supports"])
    st.session_state["s_mode"] = p["mode"]
    st.session_state["s_seg"] = p["seg"]
    if "material" in p:
        material = p["material"]
        st.session_state["s_E"] = material["E"]
        st.session_state["s_sig"] = material["sigma"]
        st.session_state["s_rho"] = material["rho"]
        st.session_state["s_amin"] = material["amin"]
        st.session_state["s_amax"] = material["amax"]
        st.session_state["s_buck"] = material["buck"]
        st.session_state["s_k"] = material["k"]
        st.session_state["s_fs"] = material["fs"]
        st.session_state["s_fam"] = material["family"]
        st.session_state["s_t"] = material["tube_ratio"]
    st.session_state["nodal0"] = pd.DataFrame(p["nodal"], columns=["Nút", "Fx (kN)", "Fy (kN)"]) if p["nodal"] else \
        pd.DataFrame({"Nút": [0], "Fx (kN)": [0.0], "Fy (kN)": [0.0]})
    st.session_state["udl0"] = pd.DataFrame(p["udl"], columns=["Thanh", "w (kN/m, hướng xuống)"]) if p["udl"] else \
        pd.DataFrame({"Thanh": [0], "w (kN/m, hướng xuống)": [0.0]})
    st.session_state["lim0"] = pd.DataFrame(p["limits"], columns=["Nút", "Phương", "|δ| max (mm)"]) if p["limits"] else \
        pd.DataFrame({"Nút": [0], "Phương": ["uy"], "|δ| max (mm)": [30.0]})
    st.session_state["s_token"] = st.session_state.get("s_token", 0) + 1
    st.session_state.pop("s_res", None)


def _init_struct_state():
    if "geo" not in st.session_state:
        _load_preset("Dàn Warren 12 m")


def build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, seg, nodal_df, udl_df, lim_df):
    nodes = [(n["x"], n["y"]) for n in geo["nodes"]]
    members = [(e["i"], e["j"]) for e in geo["elements"]]
    if len(nodes) < 2 or not members:
        raise ValueError("Cần ít nhất 2 nút và 1 thanh — hãy vẽ trên canvas hoặc chọn mẫu.")
    is_frame = mode_label.startswith("Khung")
    sup = []
    for s in geo["supports"]:
        sup.append((int(s["node"]), bool(s["ux"]), bool(s["uy"]), bool(s["rz"]) if is_frame else False))
    if not sup:
        raise ValueError("Chưa có gối tựa (chế độ 🔒 Gối tựa: bấm vào nút để đổi loại gối).")
    loads = {}
    for k, fx, fy in clean_rows(nodal_df, 3, int_cols=(0,)):
        if not 0 <= k < len(nodes):
            raise ValueError(f"Tải nút: nút {k} không tồn tại.")
        a = loads.setdefault(k, [0.0, 0.0, 0.0]); a[0] += fx; a[1] += fy
    member_q = [0.0] * len(members)
    for m, w in clean_rows(udl_df, 2, int_cols=(0,)):
        if not 0 <= m < len(members):
            raise ValueError(f"Tải phân bố: thanh {m} không tồn tại.")
        if w == 0:
            continue
        i, j = members[m]
        dx, dy = nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1]
        Lm = math.hypot(dx, dy); c, s = dx / Lm, dy / Lm
        if is_frame:
            member_q[m] += -w * c                     # thành phần vuông góc trục thanh (trục y cục bộ)
            fax = -w * s * Lm / 2.0                   # thành phần dọc trục → gộp 1/2 vào mỗi đầu thanh
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[0] += fax * c; a[1] += fax * s
        else:
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[1] += -w * Lm / 2.0
    if not any(abs(v) > 0 for a in loads.values() for v in a) and not any(member_q):
        raise ValueError("Chưa có tải trọng.")
    lims = []
    for k, d, v in [(int(r.iloc[0]), str(r.iloc[1]), float(r.iloc[2])) for _, r in lim_df.iterrows()
                    if not (pd.isna(r.iloc[0]) or pd.isna(r.iloc[1]) or pd.isna(r.iloc[2]))]:
        if not 0 <= k < len(nodes):
            raise ValueError(f"Giới hạn chuyển vị: nút {k} không tồn tại.")
        if d not in ("ux", "uy"):
            raise ValueError("Phương chuyển vị phải là ux hoặc uy.")
        if v > 0:
            lims.append((k, d, v / 1e3))
    return StructInput(nodes=nodes, members=members, supports=sup,
                       loads=[(k, a[0], a[1], a[2]) for k, a in loads.items()], member_q=member_q,
                       mode="frame" if is_frame else "truss", fam=fam, E=E, sigma_a=sig, rho=rho,
                       FS_buckling=fs, k_eff=keff, check_buckling=buck, A_min=amin, A_max=amax,
                       disp_limits=lims, segments=seg,
                       user_nodal=[(int(k), float(fx), float(fy), 0.0) for k, fx, fy in clean_rows(nodal_df, 3, int_cols=(0,))],
                       user_udl=[(int(m), float(wv)) for m, wv in clean_rows(udl_df, 2, int_cols=(0,)) if wv != 0])


def plot_structure(inp: StructInput, area_member=None, util_member=None, title="Hình học kết cấu"):
    fig = go.Figure()
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    vals = area_member
    vmin, vmax = (float(np.min(vals)), float(np.max(vals))) if vals is not None else (0, 1)
    for k, (i, j) in enumerate(inp.members):
        if vals is None:
            col, w = BLUE, 4
        else:
            t = 0.0 if vmax - vmin < 1e-12 else (vals[k] - vmin) / (vmax - vmin)
            col = pc.sample_colorscale("Turbo", 0.1 + 0.8 * t)[0]
            w = 2 + 12 * vals[k] / vmax
        fig.add_scatter(x=[inp.nodes[i][0], inp.nodes[j][0]], y=[inp.nodes[i][1], inp.nodes[j][1]], mode="lines",
                        line=dict(color=col, width=w), hoverinfo="text", showlegend=False,
                        text=f"Thanh {k}: " + (f"A = {vals[k] * 1e4:.2f} cm²" if vals is not None else ""))
        mx, my = (inp.nodes[i][0] + inp.nodes[j][0]) / 2, (inp.nodes[i][1] + inp.nodes[j][1]) / 2
        fig.add_annotation(x=mx, y=my, text=f"{k}" if vals is None else f"{vals[k] * 1e4:.1f}", showarrow=False,
                           font=dict(size=10, color="#333"), bgcolor="rgba(255,255,255,.75)")
    _add_nodes_supports_loads(fig, inp, span, xs, ys)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    return fig_layout(fig, title, 520)


def _add_nodes_supports_loads(fig, inp: StructInput, span, xs, ys):
    fig.add_scatter(x=xs, y=ys, mode="markers+text", text=[str(i) for i in range(len(xs))], textposition="top center",
                    marker=dict(size=8, color="white", line=dict(color=BLUE, width=2)), showlegend=False, name="Nút")
    sx = [inp.nodes[s[0]][0] for s in inp.supports]; sy = [inp.nodes[s[0]][1] for s in inp.supports]
    fig.add_scatter(x=sx, y=sy, mode="markers", marker=dict(symbol="triangle-up", size=14, color=RED),
                    showlegend=False, name="Gối")
    for node, fx, fy, mz in inp.loads:
        px, py = inp.nodes[node]
        mag = math.hypot(fx, fy)
        if mag > 0:
            L0 = 0.12 * span
            fig.add_annotation(x=px, y=py, ax=px - fx / mag * L0, ay=py - fy / mag * L0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowcolor=GREEN, arrowwidth=2,
                               text=f"{mag:.0f} kN", font=dict(color=GREEN, size=10))


def tab_struct():
    _init_struct_state()
    st.header("CHƯƠNG 03 · Tối ưu giàn / khung")
    tok = st.session_state["s_token"]

    top = st.columns([2, 1.3, 3])
    pre = top[0].selectbox("Mẫu có sẵn", list(PRESETS), key="s_preset")
    top[1].write(""); top[1].write("")
    top[1].button("📥 Nạp mẫu", on_click=lambda: _load_preset(st.session_state["s_preset"]), key="s_loadbtn")
    mode_label = top[2].radio("Mô hình liên kết nút", ["Dàn (nút khớp)", "Khung phẳng (nút cứng)"], horizontal=True,
                              key="s_mode")
    is_frame = mode_label.startswith("Khung")

    st.markdown("**Vẽ kết cấu** — kéo chuột để vẽ thanh; chế độ 🔒 bấm vào nút để đổi gối (ngàm → khớp → con lăn → bỏ).")
    geo = st.session_state["geo"]
    val = _canvas_component()(nodes=geo["nodes"], elements=geo["elements"], supports=geo["supports"], snap=0.5,
                              reset_token=tok, height=460, key=f"dop_canvas_{tok}", default=None)
    if val:
        geo = dict(nodes=val["nodes"], elements=val["elements"], supports=val["supports"])
        st.session_state["geo"] = geo
    st.caption(f"{len(geo['nodes'])} nút · {len(geo['elements'])} thanh · {len(geo['supports'])} gối")
    if st.session_state.get("s_preset") == "Dàn 10 thanh chuẩn (slide)":
        st.image(
            os.path.join(HERE, "assets", "ten_bar_toklu_ozbasaran.png"),
            caption="Nguồn: An efficient decoder for size optimization of trusses with automatic member grouping by Y. C. Toklu, H. Ozbasaran.",
            use_container_width=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Tải tập trung tại nút (kN)")
        nodal_df = st.data_editor(st.session_state["nodal0"], num_rows="dynamic", key=f"s_nodal_{tok}", hide_index=True,
                                  width="stretch")
    with c2:
        st.caption("Tải phân bố đứng trên thanh (kN/m theo chiều dài thanh)")
        udl_df = st.data_editor(st.session_state["udl0"], num_rows="dynamic", key=f"s_udl_{tok}", hide_index=True,
                                width="stretch")
    with c3:
        st.caption("Ràng buộc chuyển vị (mm)")
        lim_df = st.data_editor(st.session_state["lim0"], num_rows="dynamic", key=f"s_lim_{tok}", hide_index=True,
                                width="stretch",
                                column_config={"Phương": st.column_config.SelectboxColumn(options=["ux", "uy"], required=True)})

    with st.expander("⚙️ Vật liệu · họ tiết diện · ổn định", expanded=False):
        a, b, c = st.columns(3)
        st.session_state.setdefault("s_E", STEEL_E / 1e6)
        st.session_state.setdefault("s_sig", STEEL_SIGMA / 1e3)
        st.session_state.setdefault("s_rho", STEEL_RHO)
        E = a.number_input("E (GPa)", 1.0, 400.0, step=5.0, key="s_E") * 1e6
        sig = b.number_input("[σ] (MPa)", 10.0, 1000.0, step=10.0, key="s_sig") * 1e3
        rho = c.number_input("ρg (kN/m³)", 1.0, 500.0, step=0.5, key="s_rho")
        fam = family_selector("s", default="Ống tròn mỏng")
        a, b, c, d = st.columns(4)
        st.session_state.setdefault("s_amin", 2.0)
        st.session_state.setdefault("s_amax", 500.0)
        st.session_state.setdefault("s_buck", True)
        st.session_state.setdefault("s_k", 1.0)
        st.session_state.setdefault("s_fs", 1.0)
        amin = a.number_input("A_min (cm²)", 0.01, 500.0, step=0.5, key="s_amin") * 1e-4
        amax = b.number_input("A_max (cm²)", 1.0, 20000.0, step=10.0, key="s_amax") * 1e-4
        buck = c.checkbox("Ổn định Euler khi nén", key="s_buck")
        keff = d.number_input("Hệ số chiều dài k", 0.5, 2.5, step=0.1, key="s_k")
        fs = a.number_input("FS ổn định", 1.0, 3.0, step=0.1, key="s_fs")
        st.session_state.setdefault("s_seg", 4)      # giá trị gán qua session_state, không truyền value= (tránh cảnh báo)
        seg = b.number_input("Số đoạn chia / thanh (khung)", 1, 8, step=1, key="s_seg",
                             help="Chia nhỏ để A thay đổi dọc thanh (vút thon trong khung).")

    is_ten_bar = st.session_state.get("s_preset") == "Dàn 10 thanh chuẩn (slide)" and not is_frame
    with st.expander("🧬 Thuật toán cho dàn 10 thanh", expanded=is_ten_bar):
        algo = st.radio(
            "Chế độ chạy",
            ["OC/FSD + SLSQP", "GA 10 thanh", "So sánh cả hai"],
            horizontal=True,
            key="s_algorithm",
            disabled=not is_ten_bar,
            help="GA trong file toi_uu_gian_10_thanh_GA_v2.py chỉ áp dụng cho benchmark dàn 10 thanh chuẩn.",
        )
        ga_cols = st.columns(4)
        st.session_state.setdefault("ga_pop", 100)
        st.session_state.setdefault("ga_gen", 500)
        st.session_state.setdefault("ga_runs", 5)
        st.session_state.setdefault("ga_seed", 0)
        ga_pop = ga_cols[0].number_input("Quần thể", 20, 500, step=10, key="ga_pop", disabled=not is_ten_bar)
        ga_gen = ga_cols[1].number_input("Số thế hệ", 50, 2000, step=50, key="ga_gen", disabled=not is_ten_bar)
        ga_runs = ga_cols[2].number_input("Số lần chạy", 1, 20, step=1, key="ga_runs", disabled=not is_ten_bar)
        ga_seed = ga_cols[3].number_input("Seed đầu", 0, 100000, step=1, key="ga_seed", disabled=not is_ten_bar)
        ga_polish = st.checkbox("Kiểm chứng lại nghiệm GA bằng SLSQP", value=True, key="ga_polish", disabled=not is_ten_bar)
        if not is_ten_bar:
            st.caption("Chọn preset “Dàn 10 thanh chuẩn (slide)” và mô hình dàn để bật GA.")

    if st.button("🚀 Tối ưu toàn kết cấu", type="primary", key="s_go"):
        try:
            inp = build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, int(seg),
                                     nodal_df, udl_df, lim_df)
            with st.spinner("Đang tối ưu (FEM + OC + KKT)..."):
                st.session_state["s_res"] = optimize_frame(inp)
            st.session_state.pop("s_ga_res", None)
            if is_ten_bar and algo in ("GA 10 thanh", "So sánh cả hai"):
                with st.spinner("Đang chạy thuật toán di truyền GA cho dàn 10 thanh..."):
                    st.session_state["s_ga_res"] = _run_ten_bar_ga(ga_pop, ga_gen, ga_runs, ga_seed, ga_polish)
        except Exception as ex:
            st.session_state.pop("s_res", None)
            st.session_state.pop("s_ga_res", None)
            st.error(f"Lỗi: {ex}")

    res = st.session_state.get("s_res")
    if res is None:
        st.plotly_chart(_preview_fig(geo, mode_label), width="stretch")
        return
    o = res.opt
    volume_opt = float(np.sum(o.A * res.model.L))
    volume_uniform = float(o.A_uniform * np.sum(res.model.L))
    metrics([
        ("Thể tích tối ưu", f"{volume_opt:.4f} m³"),
        ("Thể tích tiết diện đều", f"{volume_uniform:.4f} m³"),
        ("Khối lượng tối ưu", f"{weight_to_mass_kg(res.weight):,.1f} kg"),
        ("Khối lượng tiết diện đều (mốc)", f"{weight_to_mass_kg(res.weight_uniform):,.1f} kg"),
        ("Tiết kiệm", f"{res.saving * 100:.1f}%", "metric-good"),
        ("U lớn nhất", f"{float(np.max(o.util)):.3f}"),
        ("Hội tụ / thoả", f"{'✔' if o.converged else '…'} / {'✔' if o.feasible else '✘'}",
         "metric-good" if o.feasible else "metric-bad"),
        ("Dừng KKT (xấp xỉ)", f"{o.EL_cv:.1%}" if o.EL_cv is not None else "—"),
    ])
    t0, tL, t2, tD, t1, t3, t4, t5 = st.tabs(["Trước / Sau", "Sơ đồ tải", "Nội lực", "Chuyển vị",
                                                "Tiết diện tối ưu", "Bảng thanh", "Hội tụ", "Mặt cắt"])
    with t0:
        st.plotly_chart(plot_structure_before_after(res), width="stretch")
        st.caption(f"Nét đứt: tiết diện đều A = {o.A_uniform * 1e4:.2f} cm² thoả cùng ràng buộc "
               f"(m = {weight_to_mass_kg(res.weight_uniform):,.1f} kg). "
               f"Nét liền: sau tối ưu (m = {weight_to_mass_kg(res.weight):,.1f} kg, "
               f"**tiết kiệm {res.saving * 100:.1f}%**). Xanh = nhỏ hơn mốc, "
                   f"cam = lớn hơn mốc. Bề rộng vẽ ∝ A^{_shape_exp(res.inp.fam):g}. Rê chuột lên giữa thanh để xem số liệu.")
    with t1:
        st.plotly_chart(plot_structure(res.inp, area_member=res.A_member, title="Diện tích A (cm²) — độ dày/màu ∝ A"),
                        width="stretch")
    with tL:
        st.plotly_chart(vz.plot_struct_loads(res), width="stretch")
        st.caption("Tải nút theo bảng nhập; với khung, tải phân bố hiển thị là thành phần vuông góc trục thanh (q⊥) mà solver dùng.")
    with t2:
        kinds = [("axial", "N — lực dọc")] if res.inp.mode == "truss" else \
            [("moment", "M — mômen"), ("shear", "V — lực cắt"), ("axial", "N — lực dọc")]
        names = [n for _, n in kinds]
        pick = st.radio("Biểu đồ nội lực", names, horizontal=True, key="s_diag") if len(kinds) > 1 else names[0]
        st.plotly_chart(vz.plot_struct_diagram(res, dict((n, k) for k, n in kinds)[pick]), width="stretch")
        if res.inp.mode == "truss":
            st.caption("Mỗi thanh có một lực dọc N gần như không đổi: đỏ = kéo (+), xanh = nén (−), "
                       "độ dày biểu thị |N|; trị số đặt giữa thanh.")
        else:
            st.caption("Biểu đồ nội lực vẽ lệch vuông góc trục thanh, có trị số tại đầu thanh và cực trị; "
                       "dùng chung tỷ lệ cho toàn kết cấu. Quy ước M dương là kéo thớ dưới theo trục cục bộ.")
    with tD:
        displacement_mode = st.segmented_control(
            "Chế độ chuyển vị",
            ["Chuyển động liên tục", "Độ võng lớn nhất"],
            default="Chuyển động liên tục",
            key="s_displacement_mode",
            label_visibility="collapsed",
            width="stretch",
        )
        if displacement_mode == "Độ võng lớn nhất":
            st.plotly_chart(vz.plot_struct_deformed(res, show_max=True), width="stretch")
        else:
            animation_html = vz.deformation_animation_html(vz.plot_struct_deformed(res))
            components.html(animation_html, height=560, scrolling=False)
        dd = res.model.node_disp(o.U)
        rows_d = [{"Nút": i, "ux (mm)": dd[i, 0] * 1e3, "uy (mm)": dd[i, 1] * 1e3,
                   **({"θz (mrad)": dd[i, 2] * 1e3} if dd.shape[1] == 3 else {})} for i in range(len(res.inp.nodes))]
        st.dataframe(pd.DataFrame(rows_d).round(4), hide_index=True, width="stretch")
    with t3:
        rows = []
        for m_, (i, j) in enumerate(res.inp.members):
            sel = res.member_of_el == m_
            Nk = float(o.N[sel][np.argmax(np.abs(o.N[sel]))])
            rows.append({"Thanh": m_, "i-j": f"{i}-{j}", "L (m)": float(np.sum(res.model.L[sel])),
                         "A (cm²)": res.A_member[m_] * 1e4, "N (kN)": Nk, "Mmax (kNm)": float(np.max(o.M[sel])),
                         "U": float(np.max(o.util[sel]))})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if res.crosscheck is not None:
            st.caption(f"Đối chiếu fem_core.solve_plane_frame: sai khác chuyển vị tương đối = {res.crosscheck['rel_diff']:.1e}")
    with t4:
        h = pd.DataFrame(o.history)
        mass_history = h["weight"].map(weight_to_mass_kg)
        ratio_max = max(float(h["max_util"].max()), float(h["max_disp_ratio"].max()), 1.0)
        fg = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Khối lượng thiết kế", "Mức sử dụng ràng buộc"),
            horizontal_spacing=0.14,
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=mass_history, row=1, col=1,
            name="Khối lượng (kg)", mode="lines+markers",
            marker=dict(size=5), line=dict(color=BLUE, width=2.5),
            hovertemplate="Vòng %{x}<br>m = %{y:.1f} kg<extra></extra>",
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=h["max_util"], row=1, col=2,
            name="Ứng suất/ổn định", mode="lines",
            line=dict(color=RED, width=2.5),
            hovertemplate="Vòng %{x}<br>U_max = %{y:.3f}<extra></extra>",
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=h["max_disp_ratio"], row=1, col=2,
            name="Chuyển vị/giới hạn", mode="lines",
            line=dict(color=ORANGE, width=2.5),
            hovertemplate="Vòng %{x}<br>δ/δ_lim = %{y:.3f}<extra></extra>",
        )
        fg.add_hline(y=1.0, row=1, col=2, line=dict(color="#455468", dash="dash", width=1.5),
                     annotation_text="Giới hạn = 1", annotation_position="top left",
                     annotation_font=dict(size=10, color="#455468"))
        fg.update_xaxes(title_text="Vòng lặp", row=1, col=1)
        fg.update_xaxes(title_text="Vòng lặp", row=1, col=2)
        fg.update_yaxes(title_text="Khối lượng (kg)", rangemode="tozero", row=1, col=1)
        fg.update_yaxes(title_text="Tỷ số (≤ 1 là đạt)", range=[0, ratio_max * 1.08], row=1, col=2)
        fg.update_layout(
            height=430,
            margin=dict(l=55, r=25, t=75, b=95),
            template="plotly_white",
            legend=dict(orientation="h", y=-0.28, yanchor="top", x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        st.plotly_chart(fg, width="stretch", config={"displayModeBar": False})
        st.caption("Đọc đồ thị: bên trái là khối lượng qua từng vòng tối ưu; bên phải là mức sử dụng lớn nhất. "
                   "Đường đỏ kiểm tra bền/ổn định, đường cam kiểm tra chuyển vị; cả hai phải ≤ 1. "
                   "Đường gạch ngang là ngưỡng giới hạn. Đây là lịch sử thuật toán, không phải biểu đồ nội lực.")
    with t5:
        member_index = st.selectbox(
            "Chọn thanh để xem mặt cắt",
            range(len(res.inp.members)),
            format_func=lambda index: f"Thanh {index}: nút {res.inp.members[index][0]}–{res.inp.members[index][1]}",
            key="s_section_member",
        )
        section_area = float(res.A_member[member_index])
        show_section_comparison(res.inp.fam, float(o.A_uniform), section_area, "s_section_comparison")
        st.caption("Diện tích hiển thị là giá trị trung bình theo chiều dài của thanh đã chọn.")
        selected_elements = np.flatnonzero(res.member_of_el == member_index)
        member_length = float(np.sum(res.model.L[selected_elements]))
        node_i, node_j = res.inp.members[member_index]
        dx = res.inp.nodes[node_j][0] - res.inp.nodes[node_i][0]
        dy = res.inp.nodes[node_j][1] - res.inp.nodes[node_i][1]
        orientation = "vertical" if abs(dy) > abs(dx) else "horizontal"
        show_section_3d(res.inp.fam, member_length, float(o.A_uniform), o.A[selected_elements],
                "s_section_3d", orientation=orientation)
    ga_res = st.session_state.get("s_ga_res")
    if ga_res is not None and is_ten_bar:
        st.subheader("So sánh hai thuật toán")
        _ten_bar_algorithm_compare(res, ga_res)
    report_block(res.report, "khung_toi_uu", "Thuyết minh — Tối ưu kết cấu khung/dàn")


def _preview_fig(geo, mode_label):
    inp = StructInput(nodes=[(n["x"], n["y"]) for n in geo["nodes"]], members=[(e["i"], e["j"]) for e in geo["elements"]],
                      supports=[(s["node"], s["ux"], s["uy"], s["rz"]) for s in geo["supports"]], loads=[], member_q=[],
                      mode="frame", fam=make_family("circle"))
    return plot_structure(inp, title="Xem trước hình học (đánh số nút/thanh)")


@st.cache_data(show_spinner=False)
def _run_ten_bar_ga(pop_size: int, generations: int, runs: int, seed0: int, use_slsqp_polish: bool):
    if any(x is None for x in (ThamSoGA, thuat_toan_di_truyen, ga_khoi_luong, ga_phan_tich_gian)):
        detail = f" Chi tiết: {GA_IMPORT_ERROR!r}" if GA_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "Không import được file toi_uu_gian_10_thanh_GA_v2.py hoặc thiếu thư viện numpy/scipy/matplotlib."
            + detail
        )
    p = ThamSoGA(kich_thuoc_quan_the=int(pop_size), so_the_he=int(generations))
    designs, histories = [], []
    for k in range(int(runs)):
        A, hist = thuat_toan_di_truyen(seed=int(seed0) + k, in_tien_trinh=False, tham_so=p)
        designs.append(A)
        histories.append(hist)
    weights_lb = np.array([ga_khoi_luong(A) for A in designs], dtype=float)
    best_idx = int(np.argmin(weights_lb))
    A_ga = np.asarray(designs[best_idx], dtype=float)
    A_sl = None
    W_sl_lb = None
    if use_slsqp_polish and toi_uu_gradient is not None:
        A_sl = np.asarray(toi_uu_gradient(A_ga), dtype=float)
        W_sl_lb = float(ga_khoi_luong(A_sl))
    u_ga, sigma_ga = ga_phan_tich_gian(A_ga)
    return {
        "A_ga": A_ga,
        "A_app": A_ga[TEN_BAR_GA_TO_APP_ORDER],
        "u_ga": np.asarray(u_ga, dtype=float),
        "sigma_ga": np.asarray(sigma_ga, dtype=float),
        "weights_lb": weights_lb,
        "histories": [np.asarray(h, dtype=float) for h in histories],
        "best_idx": best_idx,
        "W_ga_lb": float(weights_lb[best_idx]),
        "A_sl": A_sl,
        "W_sl_lb": W_sl_lb,
        "pop_size": int(pop_size),
        "generations": int(generations),
        "runs": int(runs),
        "seed0": int(seed0),
    }


def _ten_bar_algorithm_compare(res, ga_res):
    oc_lb = float(res.weight) * KN_TO_LB
    ga_lb = float(ga_res["W_ga_lb"])
    sl_lb = ga_res.get("W_sl_lb")
    diff_lb = ga_lb - oc_lb
    diff_pct = diff_lb / oc_lb * 100.0 if oc_lb else float("nan")
    better = "GA nhẹ hơn" if diff_lb < 0 else ("OC/FSD + SLSQP nhẹ hơn" if diff_lb > 0 else "Hai thuật toán bằng nhau")

    metrics([
        ("OC/FSD + SLSQP", f"{oc_lb:,.2f} lb"),
        ("GA tốt nhất", f"{ga_lb:,.2f} lb"),
        ("Chênh lệch GA - OC", f"{diff_lb:+,.2f} lb"),
        ("Chênh lệch", f"{diff_pct:+.3f}%"),
        ("Kết luận", better, "metric-good" if diff_lb <= 0 else "metric-bad"),
    ])
    if sl_lb is not None:
        st.caption(f"SLSQP kiểm chứng từ nghiệm GA: {sl_lb:,.2f} lb; lệch so với GA {sl_lb - ga_lb:+,.2f} lb "
                   f"({(sl_lb / ga_lb - 1) * 100:+.3f}%).")

    tab_summary, tab_conv, tab_shape, tab_note = st.tabs([
        "Tổng hợp chênh lệch", "Hội tụ GA", "Minh họa thiết kế GA", "Thuyết minh GA"
    ])

    with tab_summary:
        rows = []
        A_oc_in2 = np.asarray(res.A_member, dtype=float) / (0.0254 ** 2)
        sigma_ga_app = np.asarray(ga_res["sigma_ga"], dtype=float)[TEN_BAR_GA_TO_APP_ORDER]
        for i, (a_oc, a_ga, s_ga) in enumerate(zip(A_oc_in2, ga_res["A_app"], sigma_ga_app), start=1):
            rows.append({
                "Thanh": i,
                "A_OC/SLSQP (in²)": a_oc,
                "A_GA (in²)": float(a_ga),
                "Chênh A (in²)": float(a_ga - a_oc),
                "Chênh A (%)": float((a_ga / a_oc - 1.0) * 100.0) if a_oc else np.nan,
                "σ_GA (ksi)": float(s_ga),
            })
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.code(
            "\n".join([
                f"W_oc_lb = {oc_lb:.6f}",
                f"W_ga_lb = {ga_lb:.6f}",
                "chenh_lech_lb = W_ga_lb - W_oc_lb",
                "chenh_lech_pct = chenh_lech_lb / W_oc_lb * 100",
                f"chenh_lech_lb = {diff_lb:+.6f}",
                f"chenh_lech_pct = {diff_pct:+.6f}",
            ]),
            language="python",
        )

    with tab_conv:
        histories = ga_res["histories"]
        fig = go.Figure()
        for k, hist in enumerate(histories):
            y = pd.Series(hist).ffill()
            fig.add_scatter(
                x=np.arange(1, len(hist) + 1),
                y=y,
                mode="lines",
                name=f"Lần {k + 1}",
                line=dict(width=3 if k == ga_res["best_idx"] else 1.4),
                opacity=1.0 if k == ga_res["best_idx"] else 0.45,
                hovertemplate="Thế hệ %{x}<br>W = %{y:.2f} lb<extra></extra>",
            )
        fig.add_hline(y=oc_lb, line=dict(color=RED, dash="dash", width=1.5),
                      annotation_text=f"OC/FSD + SLSQP {oc_lb:.1f} lb",
                      annotation_position="top left")
        fig.add_hline(y=5060.85, line=dict(color="#111827", dash="dot", width=1.2),
                      annotation_text="Tài liệu 5060.85 lb",
                      annotation_position="bottom left")
        fig.update_layout(
            height=430,
            template="plotly_white",
            title="Quá trình hội tụ của thuật toán di truyền",
            xaxis_title="Thế hệ",
            yaxis_title="Khối lượng cá thể hợp lệ tốt nhất (lb)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
            margin=dict(l=55, r=25, t=65, b=90),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        run_rows = [{
            "Lần chạy": k + 1,
            "Seed": ga_res["seed0"] + k,
            "Khối lượng cuối (lb)": float(w),
            "Vai trò": "Tốt nhất" if k == ga_res["best_idx"] else "",
        } for k, w in enumerate(ga_res["weights_lb"])]
        st.dataframe(pd.DataFrame(run_rows).round(3), hide_index=True, width="stretch")

    with tab_shape:
        util_ga = np.abs(np.asarray(ga_res["sigma_ga"], dtype=float)[TEN_BAR_GA_TO_APP_ORDER]) / 25.0
        ga_inp = ten_bar_truss_example()
        ga_inp.fam = res.inp.fam
        st.plotly_chart(
            plot_structure(ga_inp, area_member=np.asarray(ga_res["A_app"], dtype=float) * (0.0254 ** 2),
                           util_member=util_ga,
                           title=f"Thiết kế GA — W = {ga_lb:,.2f} lb"),
            width="stretch",
        )
        st.caption("Độ dày/màu theo diện tích tiết diện GA; mức sử dụng lấy theo |σ|/[σ] của bài benchmark 10 thanh.")

    with tab_note:
        st.markdown(_ga_explanation_text(ga_res, oc_lb, ga_lb, diff_lb, diff_pct))


def _ga_explanation_text(ga_res, oc_lb, ga_lb, diff_lb, diff_pct):
    return f"""
**Thuật toán di truyền GA trong file `toi_uu_gian_10_thanh_GA_v2.py`.**

Bài toán dùng 10 biến thiết kế `A1..A10`, mỗi biến là diện tích tiết diện một thanh trong khoảng `0.1..35 in²`.
Mỗi cá thể của quần thể là một phương án thiết kế đầy đủ cho giàn 10 thanh.

**Chu trình tính toán.**

1. Khởi tạo `{ga_res["pop_size"]}` cá thể ngẫu nhiên.
2. Với từng cá thể, giải FEM để lấy chuyển vị và ứng suất.
3. Tính khối lượng `W = rho * sum(A_i * L_i)`.
4. Nếu vi phạm ứng suất hoặc chuyển vị, cộng phạt vào điểm mục tiêu.
5. Chọn lọc kiểu đấu, lai ghép SBX và đột biến đa thức để tạo thế hệ mới.
6. Giữ lại cá thể tinh hoa, lặp `{ga_res["generations"]}` thế hệ cho mỗi lần chạy.

**Cách đọc kết quả.**

- `OC/FSD + SLSQP`: thuật toán gốc của Đớp Tool, chạy tổng quát cho khung/dàn.
- `GA`: thuật toán di truyền riêng cho benchmark dàn 10 thanh.
- Công thức so sánh đang dùng:

```python
chenh_lech_lb = W_ga_lb - W_oc_lb
chenh_lech_pct = chenh_lech_lb / W_oc_lb * 100
```

Với lần chạy hiện tại: `W_oc = {oc_lb:,.2f} lb`, `W_ga = {ga_lb:,.2f} lb`,
chênh lệch `GA - OC = {diff_lb:+,.2f} lb`, tức `{diff_pct:+.3f}%`.
"""


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    inject_css()
    st.markdown(
        '<div class="course-kicker"><span class="course-code">801120</span>'
        '<span>ĐẠI HỌC TÔN ĐỨC THẮNG · KHOA KỸ THUẬT XÂY DỰNG</span></div>',
        unsafe_allow_html=True,
    )
    title_logo, title_text = st.columns([0.08, 0.92], vertical_alignment="center")
    title_logo.image(os.path.join(HERE, "assets", "dop_tool_logo.png"), width=56)
    title_text.title("Đớp Tool")
    st.caption("Tối ưu hóa kết cấu · Phép tính biến phân và FEM")
    t0, t1, t2, t3 = st.tabs([
        "01–02, 04–05 · Cơ sở & biến phân",
        "03 · Giàn / khung",
        "06 · Cột",
        "07 · Dầm",
    ])
    with t0:
        tab_theory()
    with t1:
        tab_struct()
    with t2:
        tab_column()
    with t3:
        tab_beam()


main()
