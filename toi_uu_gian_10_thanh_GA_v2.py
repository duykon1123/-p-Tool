# -*- coding: utf-8 -*-
"""
=====================================================================
 TỐI ƯU KHỐI LƯỢNG GIÀN 10 THANH BẰNG THUẬT TOÁN DI TRUYỀN (GA)
=====================================================================
Bài toán chuẩn "10-bar planar truss" (Haftka & Gürdal 1992, Lee & Geem 2004,
Camp 2007, ...) – dùng để so sánh kết quả với tài liệu.

Đơn vị: hệ Anh (in, kip, ksi, lb) như bài gốc; cuối chương trình có quy đổi SI.

Cấu trúc:
  PHẦN 1. Dữ liệu bài toán
  PHẦN 2. Phân tích giàn bằng FEM
  PHẦN 3. Hàm mục tiêu có phạt
  PHẦN 4. Thuật toán di truyền mã hóa số thực
  PHẦN 5. Kiểm chứng bằng SLSQP (gradient)
  PHẦN 6. Vẽ hình & xuất kết quả
  PHẦN 7. Chương trình chính

Cài thư viện:  pip install numpy scipy matplotlib
Chạy:          python toi_uu_gian_10_thanh_GA_v2.py
"""

from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")              # vẽ ra file ảnh, không cần mở cửa sổ
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =====================================================================
# PHẦN 1. DỮ LIỆU BÀI TOÁN
# =====================================================================
L = 360.0   # chiều dài một khoang (in)

# Tọa độ nút (x, y). Python đánh số từ 0: nút 1 trong đề = index 0.
#   nút 5 (0,360) ---- nút 3 (360,360) ---- nút 1 (720,360)
#     |                    |                    |
#   nút 6 (0,0)   ---- nút 4 (360,0)   ---- nút 2 (720,0)
NUT = np.array([
    [2 * L, L],    # nút 1
    [2 * L, 0.0],  # nút 2
    [L,     L],    # nút 3
    [L,     0.0],  # nút 4
    [0.0,   L],    # nút 5 (gối cố định)
    [0.0,   0.0],  # nút 6 (gối cố định)
])
SO_NUT = len(NUT)

# Thanh i nối (nút đầu, nút cuối) – đánh số theo đề (1..6), trừ 1 để thành index
THANH = np.array([(3, 5), (1, 3), (4, 6), (2, 4), (3, 4),
                  (1, 2), (4, 5), (3, 6), (2, 3), (1, 4)]) - 1
SO_THANH = len(THANH)

# Vật liệu và tải trọng
E = 1.0e4      # mô đun đàn hồi (ksi) – hợp kim nhôm
RHO = 0.1      # khối lượng riêng (lb/in^3)
P = 100.0      # tải trọng tập trung (kip)

# Ràng buộc thiết kế
SIGMA_CHO_PHEP = 25.0       # ứng suất cho phép kéo/nén (ksi)
CHUYEN_VI_CHO_PHEP = 2.0    # chuyển vị cho phép, mọi nút mọi phương (in)
A_MIN, A_MAX = 0.1, 35.0    # giới hạn diện tích tiết diện (in^2)


def bac_tu_do(nut, phuong):
    """Chỉ số bậc tự do: mỗi nút có 2 bậc (phương x = 0, phương y = 1)."""
    return 2 * nut + phuong


# Tải trọng: P hướng xuống (phương y, âm) tại nút 2 và nút 4
F = np.zeros(2 * SO_NUT)
F[bac_tu_do(nut=1, phuong=1)] = -P
F[bac_tu_do(nut=3, phuong=1)] = -P

# Điều kiện biên: nút 5 và 6 bị khóa cả x và y
BAC_TU_DO_KHOA = [bac_tu_do(n, p) for n in (4, 5) for p in (0, 1)]
BAC_TU_DO_TU_DO = [i for i in range(2 * SO_NUT) if i not in BAC_TU_DO_KHOA]

# Chiều dài, cos, sin của từng thanh (tính một lần, dùng nhiều lần)
_vec_thanh = NUT[THANH[:, 1]] - NUT[THANH[:, 0]]
CHIEU_DAI = np.linalg.norm(_vec_thanh, axis=1)
COS = _vec_thanh[:, 0] / CHIEU_DAI
SIN = _vec_thanh[:, 1] / CHIEU_DAI


# =====================================================================
# PHẦN 2. PHÂN TÍCH GIÀN BẰNG FEM
# =====================================================================
def ma_tran_do_cung_thanh(e, A_e):
    """Ma trận độ cứng 4x4 của thanh e (giàn 2D) trong hệ tọa độ tổng thể."""
    huong = np.array([COS[e], SIN[e], -COS[e], -SIN[e]])
    return (E * A_e / CHIEU_DAI[e]) * np.outer(huong, huong)


def phan_tich_gian(A):
    """
    Nhận vector diện tích A (10 thanh), trả về:
      u     : chuyển vị các nút (12 bậc tự do)
      sigma : ứng suất từng thanh (dương = kéo, âm = nén)
    """
    # 1) Lắp ghép ma trận độ cứng tổng thể K
    K = np.zeros((2 * SO_NUT, 2 * SO_NUT))
    for e, (i, j) in enumerate(THANH):
        chi_so = [bac_tu_do(i, 0), bac_tu_do(i, 1), bac_tu_do(j, 0), bac_tu_do(j, 1)]
        K[np.ix_(chi_so, chi_so)] += ma_tran_do_cung_thanh(e, A[e])

    # 2) Áp điều kiện biên và giải K.u = F cho các bậc tự do còn tự do
    u = np.zeros(2 * SO_NUT)
    tu_do = np.ix_(BAC_TU_DO_TU_DO, BAC_TU_DO_TU_DO)
    u[BAC_TU_DO_TU_DO] = np.linalg.solve(K[tu_do], F[BAC_TU_DO_TU_DO])

    # 3) Ứng suất: sigma = E/L * (giãn dài dọc trục thanh)
    i, j = THANH[:, 0], THANH[:, 1]
    gian_dai = ((u[2 * j] - u[2 * i]) * COS
                + (u[2 * j + 1] - u[2 * i + 1]) * SIN)
    sigma = E * gian_dai / CHIEU_DAI
    return u, sigma


def khoi_luong(A):
    """Khối lượng giàn W = rho * tổng(A_i * L_i)  (lb)."""
    return RHO * np.sum(A * CHIEU_DAI)


# =====================================================================
# PHẦN 3. HÀM MỤC TIÊU CÓ PHẠT
# =====================================================================
def vi_pham_rang_buoc(A):
    """
    Tổng mức vi phạm ràng buộc (chuẩn hóa, không thứ nguyên).
    = 0 nếu thiết kế hợp lệ; càng lớn càng vi phạm nhiều.
      g_sigma = |sigma| / sigma_cho_phep - 1  <= 0
      g_u     = |u|     / u_cho_phep     - 1  <= 0
    """
    u, sigma = phan_tich_gian(A)
    g_sigma = np.abs(sigma) / SIGMA_CHO_PHEP - 1.0
    g_u = np.abs(u) / CHUYEN_VI_CHO_PHEP - 1.0
    return np.sum(np.maximum(g_sigma, 0)) + np.sum(np.maximum(g_u, 0))


def diem_co_phat(W, vi_pham, he_so_phat):
    """Hàm mục tiêu có phạt (càng nhỏ càng tốt): f = W * (1 + he_so_phat * vi_pham)."""
    return W * (1.0 + he_so_phat * vi_pham)


# =====================================================================
# PHẦN 4. THUẬT TOÁN DI TRUYỀN MÃ HÓA SỐ THỰC
# =====================================================================
#   Mỗi "cá thể"      = một phương án thiết kế (10 giá trị diện tích)
#   Mỗi "gen"         = diện tích của một thanh
#   "Độ thích nghi"   = khối lượng có phạt (càng nhỏ càng tốt)
#   Mỗi thế hệ: ĐÁNH GIÁ -> TINH HOA -> (CHỌN LỌC -> LAI GHÉP -> ĐỘT BIẾN)
# ---------------------------------------------------------------------
@dataclass
class ThamSoGA:
    kich_thuoc_quan_the: int = 100   # số cá thể mỗi thế hệ
    so_the_he: int = 500             # số vòng lặp tiến hóa
    xac_suat_lai: float = 0.9        # xác suất lai ghép
    xac_suat_dot_bien: float = 0.1   # xác suất đột biến mỗi gen (~1/số gen)
    eta_lai: float = 15.0            # chỉ số phân bố SBX (lớn -> con gần cha mẹ)
    eta_dot_bien: float = 20.0       # chỉ số phân bố đột biến đa thức
    so_tinh_hoa: int = 2             # số cá thể tốt nhất giữ nguyên sang đời sau
    kich_thuoc_dau: int = 3          # số cá thể tham gia mỗi lần "đấu" chọn lọc
    he_so_phat_dau: float = 1.0      # hệ số phạt thế hệ đầu (cho thăm dò)
    he_so_phat_cuoi: float = 100.0   # hệ số phạt thế hệ cuối (ép về vùng hợp lệ)
    sai_so_hop_le: float = 1e-6      # vi phạm nhỏ hơn mức này coi là hợp lệ


def chon_loc_dau(quan_the, diem, k, rng):
    """CHỌN LỌC KIỂU ĐẤU (tournament): bốc ngẫu nhiên k cá thể, lấy cá thể tốt nhất."""
    ung_vien = rng.integers(0, len(quan_the), size=k)
    return quan_the[ung_vien[np.argmin(diem[ung_vien])]].copy()


def lai_ghep_SBX(cha, me, eta, rng):
    """
    LAI GHÉP SBX (Simulated Binary Crossover – Deb & Agrawal 1995):
    tạo 2 con nằm quanh cha mẹ, mô phỏng lai ghép nhị phân trên số thực.
    """
    u = rng.random(len(cha))
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    con1 = 0.5 * ((1 + beta) * cha + (1 - beta) * me)
    con2 = 0.5 * ((1 - beta) * cha + (1 + beta) * me)
    return np.clip(con1, A_MIN, A_MAX), np.clip(con2, A_MIN, A_MAX)


def do_lech_da_thuc(u, eta):
    """
    Đổi số ngẫu nhiên u (0..1) thành độ lệch delta (-1..1) của đột biến đa thức.
      u < 0,5  -> delta âm: gen bị GIẢM
      u >= 0,5 -> delta dương: gen bị TĂNG
    eta càng lớn thì delta càng gần 0, tức gen chỉ dịch chuyển nhẹ.
    Ví dụ eta = 20, u = 0,1  ->  delta ≈ -0,074  (giảm khoảng 7% khoảng biến thiên).
    """
    mu = 1 / (eta + 1)
    if u < 0.5:
        return (2 * u) ** mu - 1
    return 1 - (2 * (1 - u)) ** mu


def dot_bien_da_thuc(ca_the, pm, eta, rng):
    """
    ĐỘT BIẾN ĐA THỨC: với từng gen, làm 2 bước
      1) Bốc số ngẫu nhiên; nếu >= pm thì gen giữ nguyên (đa số trường hợp).
      2) Nếu không, bốc thêm một số u, đổi thành độ lệch delta rồi cộng vào gen:
             gen_mới = gen + delta * (A_MAX - A_MIN)
    Cuối cùng cắt lại trong [A_MIN, A_MAX]. Trả về bản sao, không sửa cá thể gốc.
    """
    ca_the = ca_the.copy()
    khoang_bien_thien = A_MAX - A_MIN
    for i in range(len(ca_the)):
        if rng.random() >= pm:      # gen này không đột biến
            continue
        delta = do_lech_da_thuc(rng.random(), eta)
        ca_the[i] += delta * khoang_bien_thien
    return np.clip(ca_the, A_MIN, A_MAX)


def sinh_the_he_moi(quan_the, diem, p, rng):
    """Tạo quần thể mới: giữ cá thể tinh hoa, phần còn lại sinh bằng chọn lọc/lai/đột biến."""
    N = p.kich_thuoc_quan_the
    thu_tu = np.argsort(diem)
    moi = [quan_the[i].copy() for i in thu_tu[:p.so_tinh_hoa]]      # TINH HOA

    while len(moi) < N:
        cha = chon_loc_dau(quan_the, diem, p.kich_thuoc_dau, rng)   # CHỌN LỌC
        me = chon_loc_dau(quan_the, diem, p.kich_thuoc_dau, rng)
        if rng.random() < p.xac_suat_lai:                           # LAI GHÉP
            cha, me = lai_ghep_SBX(cha, me, p.eta_lai, rng)
        for con in (cha, me):                                       # ĐỘT BIẾN
            if len(moi) < N:
                moi.append(dot_bien_da_thuc(con, p.xac_suat_dot_bien, p.eta_dot_bien, rng))
    return np.array(moi)


def thuat_toan_di_truyen(seed=0, in_tien_trinh=True, tham_so=None):
    """Chạy GA, trả về (thiết kế tốt nhất, lịch sử hội tụ)."""
    p = tham_so or ThamSoGA()
    rng = np.random.default_rng(seed)

    # KHỞI TẠO quần thể ngẫu nhiên trong khoảng [A_min, A_max]
    quan_the = A_MIN + rng.random((p.kich_thuoc_quan_the, SO_THANH)) * (A_MAX - A_MIN)
    lich_su = []   # khối lượng của cá thể HỢP LỆ nhẹ nhất mỗi thế hệ (nan nếu chưa có)

    for the_he in range(p.so_the_he):
        # Hệ số phạt tăng tuyến tính: đầu cho phép thăm dò vùng vi phạm, cuối ép về vùng hợp lệ
        ti_le = the_he / (p.so_the_he - 1)
        he_so_phat = p.he_so_phat_dau + (p.he_so_phat_cuoi - p.he_so_phat_dau) * ti_le

        # ĐÁNH GIÁ: mỗi cá thể chỉ phải giải FEM một lần
        W = np.array([khoi_luong(x) for x in quan_the])
        vi_pham = np.array([vi_pham_rang_buoc(x) for x in quan_the])
        diem = diem_co_phat(W, vi_pham, he_so_phat)

        # Ghi lại cá thể hợp lệ nhẹ nhất (nếu có)
        chi_so_hop_le = np.flatnonzero(vi_pham < p.sai_so_hop_le)
        if len(chi_so_hop_le) > 0:
            tot_nhat = chi_so_hop_le[np.argmin(diem[chi_so_hop_le])]
            lich_su.append(W[tot_nhat])
        else:
            lich_su.append(np.nan)
        if in_tien_trinh and (the_he % 50 == 0 or the_he == p.so_the_he - 1):
            print(f"  Thế hệ {the_he:4d}:  W tốt nhất = {lich_su[-1]:9.2f} lb")

        quan_the = sinh_the_he_moi(quan_the, diem, p, rng)

    # Kết quả cuối: cá thể có điểm phạt (hệ số phạt cuối) nhỏ nhất ở thế hệ cuối
    diem = np.array([diem_co_phat(khoi_luong(x), vi_pham_rang_buoc(x), p.he_so_phat_cuoi)
                     for x in quan_the])
    return quan_the[np.argmin(diem)], np.array(lich_su)


# =====================================================================
# PHẦN 5. KIỂM CHỨNG BẰNG PHƯƠNG PHÁP GRADIENT (SLSQP)
# =====================================================================
def toi_uu_gradient(A0):
    """Giải lại bài toán bằng SLSQP xuất phát từ A0 – kiểm tra GA có đúng vùng tối ưu không."""
    def rang_buoc(A):   # dạng g >= 0 theo yêu cầu của SciPy
        u, sigma = phan_tich_gian(A)
        return np.concatenate([1 - np.abs(sigma) / SIGMA_CHO_PHEP,
                               1 - np.abs(u[BAC_TU_DO_TU_DO]) / CHUYEN_VI_CHO_PHEP])

    kq = minimize(khoi_luong, A0, method="SLSQP",
                  bounds=[(A_MIN, A_MAX)] * SO_THANH,
                  constraints={"type": "ineq", "fun": rang_buoc},
                  options={"maxiter": 500, "ftol": 1e-10})
    return kq.x


def thiet_ke_dong_deu():
    """Thiết kế truyền thống để so sánh: mọi thanh cùng tiết diện, chọn nhỏ nhất vẫn thỏa ràng buộc."""
    for a in np.arange(A_MIN, A_MAX, 0.01):
        if vi_pham_rang_buoc(np.full(SO_THANH, a)) < 1e-9:
            return np.full(SO_THANH, a)
    return np.full(SO_THANH, A_MIN)


# =====================================================================
# PHẦN 6. VẼ HÌNH & XUẤT KẾT QUẢ
# =====================================================================
def ve_gian(A, tieu_de, ten_file, sigma):
    """Vẽ giàn: độ dày nét tỉ lệ với diện tích; đỏ = kéo, xanh = nén."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Thanh + nhãn "số hiệu / diện tích"
    for e, (i, j) in enumerate(THANH):
        mau = "#c0392b" if sigma[e] > 0 else "#2c6fbb"
        ax.plot(NUT[[i, j], 0], NUT[[i, j], 1], color=mau,
                linewidth=1 + 14 * A[e] / A_MAX, solid_capstyle="round", zorder=1)
        # Thanh xiên cắt nhau ở giữa -> đặt nhãn ở 1/4 chiều dài để không chồng nhau
        t = 0.25 if (COS[e] != 0 and SIN[e] != 0) else 0.5
        xm, ym = NUT[i] + t * (NUT[j] - NUT[i])
        ax.text(xm, ym, f"{e+1}\n{A[e]:.2f}", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=3)

    # Nút
    ax.scatter(NUT[:, 0], NUT[:, 1], s=40, color="black", zorder=2)
    for k, (x, y) in enumerate(NUT):
        ax.text(x + 12, y + 14, f"({k+1})", fontsize=9)

    # Tải trọng (nút 2 và 4) và gối tựa (nút 5 và 6)
    for k in (1, 3):
        ax.annotate("", xy=NUT[k], xytext=NUT[k] + [0, -110],
                    arrowprops=dict(arrowstyle="<-", color="#e67e22", lw=2))
        ax.text(NUT[k, 0] + 10, NUT[k, 1] - 100, f"P = {P:.0f} kip", color="#e67e22", fontsize=9)
    for k in (4, 5):
        ax.plot(NUT[k, 0] - 18, NUT[k, 1], marker=">", markersize=14, color="black")

    ax.plot([], [], color="#c0392b", lw=4, label="Thanh chịu kéo")
    ax.plot([], [], color="#2c6fbb", lw=4, label="Thanh chịu nén")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(tieu_de + "\n(số trên thanh: số hiệu / diện tích in², nét dày ∝ diện tích)",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-80, 800)
    ax.set_ylim(-170, 440)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ten_file, dpi=200)
    plt.close(fig)


def ve_hoi_tu(lich_su_tat_ca, chi_so_tot_nhat, kq_tai_lieu, ten_file="hoi_tu_GA.png"):
    """Vẽ biểu đồ hội tụ của các lần chạy GA (lần tốt nhất vẽ đậm)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, ls in enumerate(lich_su_tat_ca):
        tot = (k == chi_so_tot_nhat)
        ax.plot(ls, lw=1.6 if tot else 0.9, alpha=1 if tot else 0.45, label=f"Lần chạy {k+1}")
    ax.axhline(kq_tai_lieu, color="black", ls="--", lw=1,
               label=f"Kết quả công bố {kq_tai_lieu} lb")
    ax.set_xlabel("Thế hệ")
    ax.set_ylabel("Khối lượng tốt nhất hợp lệ (lb)")
    ax.set_title("Biểu đồ hội tụ của thuật toán di truyền")
    ax.set_ylim(4900, 8000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ten_file, dpi=200)
    plt.close(fig)


def luu_csv(A_ga, s_ga, A_sl, s_sl, ten_file="ket_qua.csv"):
    """Lưu số liệu ra CSV (utf-8-sig để Excel mở không lỗi font) cho báo cáo."""
    with open(ten_file, "w", encoding="utf-8-sig") as f:
        f.write("Thanh,A_GA (in2),Ung suat GA (ksi),A_SLSQP (in2),Ung suat SLSQP (ksi)\n")
        for e in range(SO_THANH):
            f.write(f"{e+1},{A_ga[e]:.3f},{s_ga[e]:.2f},{A_sl[e]:.3f},{s_sl[e]:.2f}\n")
        f.write(f"Khoi luong (lb),{khoi_luong(A_ga):.2f},,{khoi_luong(A_sl):.2f},\n")


def in_ket_qua(A_ga, A_sl, W_cac_lan, A_deu, kq_tai_lieu):
    """In bảng kết quả và so sánh ra màn hình."""
    u_ga, s_ga = phan_tich_gian(A_ga)
    W_ga = W_cac_lan.min()
    W_deu = khoi_luong(A_deu)

    print("\n" + "=" * 62)
    print(" KẾT QUẢ")
    print("=" * 62)
    print(f"{'Thanh':>5} | {'A_GA (in²)':>10} | {'σ_GA (ksi)':>10} | {'A_SLSQP':>8}")
    for e in range(SO_THANH):
        print(f"{e+1:>5} | {A_ga[e]:10.3f} | {s_ga[e]:10.2f} | {A_sl[e]:8.3f}")
    print("-" * 62)
    print(f"Khối lượng GA (tốt nhất / TB / xấu nhất qua {len(W_cac_lan)} lần):")
    print(f"   {W_ga:.2f} / {W_cac_lan.mean():.2f} / {W_cac_lan.max():.2f} lb"
          f"   (độ lệch chuẩn {W_cac_lan.std():.2f})")
    print(f"Khối lượng SLSQP (kiểm chứng): {khoi_luong(A_sl):.2f} lb")
    print(f"Khối lượng công bố:            {kq_tai_lieu:.2f} lb")
    print(f"Sai lệch GA so với tài liệu:   {(W_ga / kq_tai_lieu - 1) * 100:+.2f} %")
    print(f"Chuyển vị lớn nhất (GA): {np.abs(u_ga).max():.3f} in  (giới hạn {CHUYEN_VI_CHO_PHEP})")
    print(f"Ứng suất lớn nhất  (GA): {np.abs(s_ga).max():.2f} ksi (giới hạn {SIGMA_CHO_PHEP:.0f})")
    print(f"\nThiết kế đồng đều (mọi thanh A = {A_deu[0]:.2f} in²): {W_deu:.2f} lb")
    print(f"=> Tối ưu giảm {(1 - W_ga / W_deu) * 100:.1f} % khối lượng")
    print(f"\nQuy đổi SI: W_GA = {W_ga * 0.45359:.1f} kg; "
          f"1 in² = 6.4516 cm²; 1 ksi = 6.895 MPa")


# =====================================================================
# PHẦN 7. CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def main():
    KQ_TAI_LIEU = 5060.85   # lb – nghiệm tối ưu công bố (Haftka & Gürdal 1992)
    SO_LAN_CHAY = 5         # GA là ngẫu nhiên -> chạy nhiều lần để đánh giá

    print("=" * 62)
    print(" TỐI ƯU GIÀN 10 THANH BẰNG THUẬT TOÁN DI TRUYỀN")
    print("=" * 62)

    # 1) Chạy GA nhiều lần (mỗi lần một seed), lấy lần nhẹ nhất
    ket_qua, lich_su_tat_ca = [], []
    for lan in range(SO_LAN_CHAY):
        print(f"\nLần chạy {lan + 1}/{SO_LAN_CHAY} (seed = {lan}):")
        A, ls = thuat_toan_di_truyen(seed=lan)
        ket_qua.append(A)
        lich_su_tat_ca.append(ls)

    W_cac_lan = np.array([khoi_luong(a) for a in ket_qua])
    chi_so_tot = int(np.argmin(W_cac_lan))
    A_ga = ket_qua[chi_so_tot]

    # 2) Kiểm chứng bằng gradient (xuất phát từ nghiệm GA) và so với thiết kế đồng đều
    A_sl = toi_uu_gradient(A_ga)
    A_deu = thiet_ke_dong_deu()

    # 3) In kết quả
    in_ket_qua(A_ga, A_sl, W_cac_lan, A_deu, KQ_TAI_LIEU)

    # 4) Vẽ hình và lưu số liệu
    _, s_ga = phan_tich_gian(A_ga)
    _, s_sl = phan_tich_gian(A_sl)
    _, s_deu = phan_tich_gian(A_deu)

    ve_hoi_tu(lich_su_tat_ca, chi_so_tot, KQ_TAI_LIEU)
    ve_gian(A_deu, f"Thiết kế đồng đều – W = {khoi_luong(A_deu):.0f} lb",
            "gian_ban_dau.png", s_deu)
    ve_gian(A_ga, f"Thiết kế tối ưu bằng GA – W = {W_cac_lan.min():.0f} lb",
            "gian_toi_uu.png", s_ga)
    print("\nĐã lưu hình: hoi_tu_GA.png, gian_ban_dau.png, gian_toi_uu.png")

    luu_csv(A_ga, s_ga, A_sl, s_sl)
    print("Đã lưu số liệu: ket_qua.csv")


if __name__ == "__main__":
    main()
