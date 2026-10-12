"""
opt_core.py — Lõi tối ưu hoá kết cấu thép cho "Đớp Tool"

Hai tầng:
  A. Phép tính biến phân (Calculus of Variations) — biến thiết kế là hàm liên tục A(x), I(x)
       * Cột chịu nén (Lagrange / Keller / Tadjbakhsh): phương trình Euler–Lagrange dạng ODE
         -> nghiệm đóng qua hàm Beta không đầy đủ.
       * Dầm chịu uốn (min thể tích, ràng buộc độ võng): F không chứa A' -> Euler–Lagrange
         suy biến thành phương trình ĐẠI SỐ:  A^(n+1) = λ·n·M·m̄ / (E·α).
  B. Tối ưu số trên lưới FEM (tiêu chuẩn tối ưu OC + Fully-Stressed-Design)
       * Phân tích nội lực/chuyển vị dùng lại ma trận phần tử của fem_core.py (không sửa core).
       * Dùng cho dầm siêu tĩnh, khung phẳng và dàn.

Đơn vị: kN, m, kN/m²  (E thép ≈ 2.1e8 kN/m², ứng suất 1 MPa = 1e3 kN/m²).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.special import beta as beta_fn
from scipy.special import betaincinv

from fem_core import (
    FrameElement, FrameNode, FramePointLoad, FrameSupport, PlaneFrameInput,
    _beam_element_stiffness,
    _frame_element_stiffness_local,
    _frame_internal_force_at,
    _frame_udl_local_to_global,
    _rotation_matrix,
    solve_plane_frame,
)

STEEL_E = 2.1e8      # kN/m²
STEEL_RHO = 78.5     # kN/m³ (trọng lượng riêng)
STEEL_SIGMA = 2.1e5  # kN/m² (210 MPa)
GRAVITY_STANDARD = 9.80665  # m/s²


def weight_to_mass_kg(weight_kN: float) -> float:
    """Convert a force in kN to its equivalent mass in kg using standard gravity."""
    return float(weight_kN) * 1000.0 / GRAVITY_STANDARD


def two_bar_kkt_solution(P_kN: float, B_m: float, sigma_allow_kN_m2: float,
                         rho_kg_m3: float) -> dict:
    """Closed-form symmetric two-bar design and its KKT checks (lecture example)."""
    values = (P_kN, B_m, sigma_allow_kN_m2, rho_kg_m3)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("P, B, ứng suất cho phép và khối lượng riêng phải dương, hữu hạn.")

    h = B_m
    alpha = math.pi / 4
    length = math.hypot(B_m, h)
    area = P_kN * length / (2 * sigma_allow_kN_m2 * h)
    volume = 2 * area * length
    axial_force = P_kN / (2 * math.sin(alpha))
    stress = axial_force / area
    lagrange_multiplier = 2 * P_kN * B_m / sigma_allow_kN_m2**2
    g = stress - sigma_allow_kN_m2
    stationarity_A = 2 * length - lagrange_multiplier * P_kN * length / (2 * area**2 * h)
    stationarity_h = (2 * area * h / length
                      - lagrange_multiplier * P_kN * B_m**2 / (2 * area * length * h**2))
    complementarity = lagrange_multiplier * g
    sigma_mpa = sigma_allow_kN_m2 / 1e3
    stress_mpa = stress / 1e3
    lambda_from_A = 4 * area**2 * h / P_kN
    lambda_from_h = 4 * area**2 * h**3 / (P_kN * B_m**2)
    volume_from_closed_form = 2 * P_kN * B_m / sigma_allow_kN_m2
    mass = rho_kg_m3 * volume
    tolerance = 1e-9 * max(1.0, sigma_allow_kN_m2)
    report = "\n".join([
        "=" * 56,
        "   THUYẾT MINH — BÀI 3: GIÀN HAI THANH ĐỐI XỨNG",
        "=" * 56,
        "",
        "1. DỮ LIỆU, GIẢ THIẾT VÀ BIẾN THIẾT KẾ",
        f"   Khoảng cách hai gối: 2B = {2 * B_m:.4f} m; tải đứng P = {P_kN:.4f} kN.",
        f"   [σ] = {sigma_mpa:.4f} MPa = {sigma_allow_kN_m2:.4f} kN/m²; ρ = {rho_kg_m3:.2f} kg/m³.",
        "   Hai thanh giống nhau, đàn hồi tuyến tính; chỉ xét điều kiện bền kéo/nén, chưa xét Euler.",
        "   Biến thiết kế: chiều cao h > 0 và diện tích mỗi thanh A > 0.",
        "",
        "2. THIẾT LẬP MÔ HÌNH HÌNH HỌC VÀ NỘI LỰC",
        "   2.1 Hình học: l = √(B² + h²); sin(α) = h/l; cos(α) = B/l.",
        f"       Thay nghiệm: l* = √({B_m:.4f}² + {h:.4f}²) = {length:.6f} m; α* = 45°.",
        "   2.2 Cân bằng nút đỉnh: 2N sin(α) = P.",
        f"       N = P/(2 sin α) = {P_kN:.4f}/(2 sin 45°) = {axial_force:.6f} kN.",
        "       σ = N/A = Pl/(2Ah); điều kiện bền: g(h,A) = Pl/(2Ah) − [σ] ≤ 0.",
        "",
        "3. HÀM MỤC TIÊU VÀ KHỬ RÀNG BUỘC HOẠT ĐỘNG",
        "   Thể tích: V(h,A) = 2Al = 2A√(B² + h²) → min.",
        "   Vì V tăng theo A nên tại tối ưu ràng buộc bền hoạt động, g(h,A) = 0.",
        "   Giải g = 0 theo A: A(h) = Pl/(2[σ]h).",
        "   Thế vào thể tích và dùng l² = B² + h²:",
        "       V(h) = Pl²/([σ]h) = P/[σ] · (B²/h + h).",
        "",
        "4. TÌM CỰC TIỂU — CÁCH 1: ĐẠO HÀM THEO h",
        "   dV/dh = P/[σ] · (1 − B²/h²) = 0  ⇒  h² = B².",
        f"       Do h > 0: h* = B = {h:.6f} m.",
        "   d²V/dh² = 2PB²/([σ]h³) > 0 với mọi h > 0.",
        "       Do đó điểm dừng là cực tiểu nghiêm ngặt; V(h) lồi trên miền h > 0.",
        "",
        "5. KIỂM TRA HÌNH HỌC — CÁCH 2: BIẾN ĐỔI SANG GÓC α",
        "   h = B tan(α), l = B/cos(α).",
        "   V(α) = 2PB/([σ] sin(2α)); V nhỏ nhất khi sin(2α) lớn nhất.",
        "   sin(2α) ≤ 1 ⇒ α* = 45° ⇒ h* = B.",
        "   Hai cách độc lập cùng cho h* = B; đây là bước đối chiếu nghiệm giải tích.",
        "",
        "6. THẾ SỐ KẾT QUẢ THIẾT KẾ",
        f"   h* = B = {h:.6f} m = {h * 1000:.1f} mm; α* = 45°; l* = √2B = {length:.6f} m.",
        f"   A* = Pl*/(2[σ]h*) = {P_kN:.4f}·{length:.6f}/(2·{sigma_allow_kN_m2:.4f}·{h:.6f})",
        f"      = {area:.8e} m² = {area * 1e6:.3f} mm² mỗi thanh.",
        f"   V* = 2A*l* = 2·{area:.8e}·{length:.6f} = {volume:.8e} m³.",
        f"   m* = ρV* = {rho_kg_m3:.2f}·{volume:.8e} = {mass:.4f} kg.",
        f"   Kiểm tra bền: σ* = N/A* = {axial_force:.6f}/{area:.8e} = {stress_mpa:.4f} MPa = [σ].",
        "",
        "7. KIỂM CHỨNG NGHIỆM BẰNG ĐIỀU KIỆN KKT",
        "   7.1 Lagrangian cho g(h,A) ≤ 0:",
        "       ℒ = 2A√(B²+h²) + λ(Pl/(2Ah) − [σ]),  λ ≥ 0.",
        "   7.2 Điều kiện dừng theo A:",
        "       ∂ℒ/∂A = 2l − λPl/(2A²h) = 0  ⇒  λ_A = 4A²h/P.",
        f"       λ_A(thế số nghiệm) = {lambda_from_A:.8e} m⁵/kN.",
        "   7.3 Điều kiện dừng theo h; dùng d(l/h)/dh = −B²/(l h²):",
        "       ∂ℒ/∂h = 2Ah/l − λPB²/(2Alh²) = 0  ⇒  λ_h = 4A²h³/(PB²).",
        f"       λ_h(thế số nghiệm) = {lambda_from_h:.8e} m⁵/kN.",
        "   7.4 Đồng nhất λ_A = λ_h, khử các đại lượng dương 4A²h/P:",
        "       1 = h²/B²  ⇒  h* = B (vì h > 0), trùng với hai cách giải ở trên.",
        f"       λ* = λ_A = λ_h = 2PB/[σ]² = {lagrange_multiplier:.8e} m⁵/kN.",
        "   7.5 Bảng kiểm tra đầy đủ các điều kiện KKT:",
        f"       Dừng: |∂ℒ/∂A| = {abs(stationarity_A):.3e}; |∂ℒ/∂h| = {abs(stationarity_h):.3e}.",
        f"       Khả thi nguyên thủy: g = {g:.3e} kN/m² ≤ 0  — {'ĐẠT' if g <= tolerance else 'KHÔNG ĐẠT'}.",
        f"       Khả thi đối ngẫu: λ* = {lagrange_multiplier:.8e} ≥ 0  — {'ĐẠT' if lagrange_multiplier >= 0 else 'KHÔNG ĐẠT'}.",
        f"       Bù trừ: λ*g = {complementarity:.3e}  — {'ĐẠT' if abs(complementarity) <= tolerance else 'KHÔNG ĐẠT'}.",
        "       Biên h > 0, A > 0 không hoạt động tại nghiệm; nhân tử biên tương ứng bằng 0.",
        "       Tất cả điều kiện KKT được thỏa; đạo hàm bậc hai d²V/dh² > 0 xác nhận cực tiểu toàn cục của bài toán đã khử còn một biến.",
        "",
        "8. Ý NGHĨA NHÂN TỬ LAGRANGE VÀ KẾT LUẬN",
        "   λ* là giá bóng của ràng buộc ứng suất cho phép:",
        f"       λ* = −dV*/d[σ] = 2PB/[σ]² = V*/[σ] = {volume_from_closed_form / sigma_allow_kN_m2:.8e} m⁵/kN.",
        "   Nới [σ] một lượng nhỏ d[σ] làm thể tích tối ưu đổi xấp xỉ dV* = −λ* d[σ].",
        f"   Kết luận: h* = {h * 1000:.1f} mm; α* = 45°; A* = {area * 1e6:.3f} mm²/thanh; "
        f"V* = {volume:.6e} m³; m* = {mass:.3f} kg.",
        "   Phạm vi: kết quả chỉ xét bền kéo/nén như bài tập; cấu kiện chịu nén thực tế cần kiểm tra thêm ổn định Euler.",
    ])
    return {
        "h": h,
        "alpha": alpha,
        "length": length,
        "area": area,
        "volume": volume,
        "mass": rho_kg_m3 * volume,
        "axial_force": axial_force,
        "stress": stress,
        "lagrange_multiplier": lagrange_multiplier,
        "stationarity_A": stationarity_A,
        "stationarity_h": stationarity_h,
        "constraint": g,
        "complementarity": complementarity,
        "report": report,
    }


def lecture_examples() -> dict[str, dict]:
    """Bộ ví dụ slide chuẩn cho toàn bộ chuỗi giảng dạy: giàn, cột, dầm."""
    truss = two_bar_kkt_solution(150.0, 1.0, 180e3, 7850.0)
    fam_col = make_family("circle")
    col = optimize_column(5.0, fam_col, STEEL_E, "pinned", V=5.0 * 1e-3, run_numeric=False)

    fam_beam = make_family("ibeam", h=0.4)
    beam_prob = BeamProblem(
        L=8.0,
        bc="ss",
        udls=[(20.0, 0.0, 8.0)],
        point_loads=[(100.0, 4.0)],
        fam=fam_beam,
        E=STEEL_E,
        sigma_a=1.0e12,
        delta_max=0.02,
        x_ctrl=4.0,
        A_min=2e-4,
        n_elem=80,
    )
    beam = optimize_beam(beam_prob)

    return {
        "two_bar": {
            "title": "Giàn hai thanh đối xứng",
            "summary": "Tối ưu khối lượng giàn với ràng buộc bền kéo/nén và kiểm tra KKT.",
            "report": truss["report"],
            "result": truss,
            "geometry": {"B": 1.0, "P_kN": 150.0},
        },
        "column": {
            "title": "Cột chịu nén biến tiết diện",
            "summary": "Cực đại tải tới hạn theo Keller, so sánh nghiệm đóng và FEM.",
            "report": col.report,
            "result": col,
        },
        "beam": {
            "title": "Dầm vút thon theo độ võng cho trước",
            "summary": "Tối ưu A(x) với ràng buộc độ võng và bền bằng biến phân + FEM.",
            "report": beam.report,
            "result": beam,
        },
    }


def ten_bar_truss_example() -> StructInput:
    """Classic 10-member, two-panel X-braced truss shown in the lecture slide.

    Geometry and loads are defined in the source problem's in, kip, ksi units,
    then converted to the application's kN, m, kN/m² units.
    """
    inch_to_m = 0.0254
    kip_to_kN = 4.4482216152605
    ksi_to_kN_m2 = 6894.757293168
    inch2_to_m2 = inch_to_m**2
    nodes_in = [(0, 360), (360, 360), (720, 360), (0, 0), (360, 0), (720, 0)]
    members = [
        (0, 1), (1, 2), (3, 4), (4, 5), (1, 4),
        (2, 5), (3, 1), (0, 4), (4, 2), (1, 5),
    ]
    nodes = [(x * inch_to_m, y * inch_to_m) for x, y in nodes_in]
    loads = [(4, 0.0, -100.0 * kip_to_kN, 0.0),
             (5, 0.0, -100.0 * kip_to_kN, 0.0)]
    free_nodes = (1, 2, 4, 5)
    displacement_limit = 2.0 * inch_to_m
    density_kN_m3 = (0.1 * 4.4482216152605 / inch_to_m**3) / 1000.0

    return StructInput(
        nodes=nodes,
        members=members,
        supports=[(0, True, True, False), (3, True, True, False)],
        loads=loads,
        member_q=[0.0] * len(members),
        mode="truss",
        fam=make_family("tube", ratio=0.05),
        E=10_000.0 * ksi_to_kN_m2,
        sigma_a=25.0 * ksi_to_kN_m2,
        rho=density_kN_m3,
        check_buckling=False,
        A_min=0.1 * inch2_to_m2,
        A_max=1.0,
        disp_limits=[(node, component, displacement_limit)
                     for node in free_nodes for component in ("ux", "uy")],
        segments=1,
        user_nodal=list(loads),
    )


# ══════════════════════════════════════════════════════════════
#  1. HỌ TIẾT DIỆN   I = α·A^n ,  W = β·A^(n-γ)   (độ cao c ∝ A^γ)
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SectionFamily:
    name: str
    n: float
    alpha: float
    gamma: float
    beta: float
    desc: str = ""
    shape_kind: str = "custom"
    shape_parameters: tuple[float, ...] = ()

    def I(self, A):
        return self.alpha * np.power(A, self.n)

    def W(self, A):
        return self.beta * np.power(A, self.n - self.gamma)

    def dI_dA(self, A):
        return self.n * self.alpha * np.power(A, self.n - 1.0)

    def A_from_W(self, W):
        return np.power(np.maximum(W, 0.0) / self.beta, 1.0 / (self.n - self.gamma))

    def A_from_I(self, I):
        return np.power(np.maximum(I, 0.0) / self.alpha, 1.0 / self.n)

    def cross_section_dimensions(self, area: float) -> dict[str, float]:
        """Return model-consistent cross-section dimensions in meters."""
        area = float(area)
        if not math.isfinite(area) or area <= 0:
            raise ValueError("Diện tích tiết diện phải dương và hữu hạn.")

        if self.shape_kind == "rect":
            ratio = self.shape_parameters[0]
            height = math.sqrt(area / ratio)
            return {"b": ratio * height, "h": height}
        if self.shape_kind == "circle":
            return {"D": math.sqrt(4 * area / math.pi)}
        if self.shape_kind == "tube":
            ratio = self.shape_parameters[0]
            diameter = math.sqrt(area / (math.pi * ratio))
            return {"D": diameter, "t": ratio * diameter}
        if self.shape_kind == "ibeam":
            height = self.shape_parameters[0]
            flange_thickness = 0.08 * height
            return {"h": height, "b_f": area / (2 * flange_thickness), "t_f": flange_thickness}
        return {"A": area}


def make_family(kind: str, **kw) -> SectionFamily:
    """Tạo họ tiết diện.
    kind: 'rect' (ratio=b/h) | 'circle' | 'tube' (ratio=t/D) | 'ibeam' (h) | 'custom' (n, alpha, gamma, beta)
    """
    if kind == "rect":
        r = float(kw.get("ratio", 0.5))
        return SectionFamily("Chữ nhật đặc đồng dạng (b/h=%.2f)" % r, 2.0, 1.0 / (12.0 * r), 0.5,
                             1.0 / (6.0 * math.sqrt(r)),
                             "Mọi kích thước tỷ lệ với nhau: I ∝ A²", "rect", (r,))
    if kind == "circle":
        return SectionFamily("Tròn đặc", 2.0, 1.0 / (4.0 * math.pi), 0.5,
                             1.0 / (4.0 * math.sqrt(math.pi)), "I ∝ A²", "circle")
    if kind == "tube":
        r = float(kw.get("ratio", 0.05))
        return SectionFamily("Ống tròn mỏng đồng dạng (t/D=%.3f)" % r, 2.0,
                             1.0 / (8.0 * math.pi * r), 0.5,
                             1.0 / (4.0 * math.sqrt(math.pi * r)),
                             "Ống thành mỏng, t/D cố định: I ∝ A²", "tube", (r,))
    if kind == "ibeam":
        h = float(kw.get("h", 0.3))
        return SectionFamily("Chữ I cánh mỏng, h=%.3f m cố định" % h, 1.0, h * h / 4.0, 0.0, h / 2.0,
                             "A = diện tích hai cánh, bỏ qua bụng: I = A·h²/4, n = 1", "ibeam", (h,))
    if kind == "custom":
        n = float(kw["n"]); g = float(kw["gamma"])
        return SectionFamily("Tuỳ chỉnh (n=%.2f)" % n, n, float(kw["alpha"]), g, float(kw["beta"]), "",
                             "custom")
    raise ValueError(f"Họ tiết diện không hợp lệ: {kind}")


# ══════════════════════════════════════════════════════════════
#  2. CỘT — EULER–LAGRANGE DẠNG ODE, NGHIỆM ĐÓNG
# ══════════════════════════════════════════════════════════════
#  Cực đại tải tới hạn P với thể tích V = ∫A dx cho trước, I = α·A^n.
#  Rayleigh: P = min ∫EI w''² dx / ∫w'² dx . Điều kiện dừng theo A:
#       n·E·α·A^(n-1)·w''² = λ   (đều)       (*)
#  Kết hợp M = EI·w'' = -P·w  ⇒  A ∝ |w|^(2/(n+1))  và  w'' = -κ·|w|^((1-n)/(n+1))·sign(w).
#  Tích phân lần đầu + đổi biến u = A/Amax cho  ξ = I_u((n+1)/2, 1/2)  (Beta không đầy đủ chuẩn hoá)
#  ⇒  u(ξ) = betaincinv((n+1)/2, 1/2, ξ).
#  Hệ quả:  <A>/Amax = (n+1)/(n+2) ;   P = (n+1)·B((n+1)/2,1/2)²·E·α·Amax^n / L_ef².

COLUMN_BCS = {
    "pinned": "Hai đầu khớp",
    "cantilever": "Một đầu ngàm (x=0) – một đầu tự do (x=L)",
    "clamped": "Hai đầu ngàm",
}


def _column_xi(bc: str, t: np.ndarray) -> np.ndarray:
    """ξ ∈ [0,1] (0: tiết diện triệt tiêu, 1: tiết diện lớn nhất) theo toạ độ chuẩn hoá t=x/L."""
    t = np.asarray(t, float)
    if bc == "pinned":
        return 1.0 - np.abs(1.0 - 2.0 * t)
    if bc == "cantilever":
        return 1.0 - t                       # ngàm ở x=0 (A max), đầu tự do ở x=L (A→0)
    if bc == "clamped":
        tt = np.minimum(t, 1.0 - t)
        return np.abs(1.0 - 4.0 * tt)        # A→0 tại L/4 và 3L/4
    raise ValueError(bc)


def _column_len_factor(bc: str) -> float:
    """Chiều dài hiệu dụng / L."""
    return {"pinned": 1.0, "cantilever": 2.0, "clamped": 0.5}[bc]


def column_profile(n: float, bc: str, L: float, V: float, x: np.ndarray) -> np.ndarray:
    """A(x) tối ưu dạng đóng (có thể bằng 0 ở các điểm đặc biệt)."""
    a = (n + 1.0) / 2.0
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    xi = np.clip(_column_xi(bc, np.asarray(x) / L), 0.0, 1.0)
    return Amax * betaincinv(a, 0.5, xi)


def column_critical_load_closed(n: float, alpha: float, E: float, L: float, V: float, bc: str) -> float:
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    B = beta_fn((n + 1.0) / 2.0, 0.5)
    Lef = _column_len_factor(bc) * L
    return (n + 1.0) * B ** 2 * E * alpha * Amax ** n / Lef ** 2


def column_uniform_load(n: float, alpha: float, E: float, L: float, V: float, bc: str) -> float:
    A0 = V / L
    Lef = _column_len_factor(bc) * L
    return math.pi ** 2 * E * alpha * A0 ** n / Lef ** 2


def _geo_stiffness(Le: float) -> np.ndarray:
    return 1.0 / (30.0 * Le) * np.array([
        [36, 3 * Le, -36, 3 * Le],
        [3 * Le, 4 * Le ** 2, -3 * Le, -Le ** 2],
        [-36, -3 * Le, 36, -3 * Le],
        [3 * Le, -Le ** 2, -3 * Le, 4 * Le ** 2],
    ])


def column_buckling_fem(L_el: np.ndarray, EI_el: np.ndarray, bc: str):
    """Tải tới hạn bằng FEM (ma trận độ cứng Hermite của fem_core + ma trận độ cứng hình học).
    Gốc x=0 là đầu ngàm với 'cantilever'. Trả (P_cr, vector chuyển vị đầy đủ)."""
    N = len(EI_el)
    nd = 2 * (N + 1)
    K = np.zeros((nd, nd)); G = np.zeros((nd, nd))
    for e in range(N):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += _beam_element_stiffness(float(EI_el[e]), float(L_el[e]))
        G[np.ix_(d, d)] += _geo_stiffness(float(L_el[e]))
    if bc == "pinned":
        fixed = [0, 2 * N]
    elif bc == "cantilever":
        fixed = [0, 1]
    elif bc == "clamped":
        fixed = [0, 1, 2 * N, 2 * N + 1]
    else:
        raise ValueError(bc)
    free = [d for d in range(nd) if d not in fixed]
    w, V = eigh(K[np.ix_(free, free)], G[np.ix_(free, free)], subset_by_index=[0, 0])
    U = np.zeros(nd); U[free] = V[:, 0]
    return float(w[0]), U


def column_bimodal_slsqp(L: float, V: float, fam: SectionFamily, E: float, bc: str,
                         N: int = 60, n_modes: int = 3, A_init: np.ndarray | None = None):
    """Cực đại min(λ1, λ2, λ3) (bài toán đa trị riêng / bimodal) với thể tích cố định bằng SLSQP + đạo hàm riêng.
    Cần cho cột HAI ĐẦU NGÀM: nghiệm đơn-mode Euler–Lagrange (A=0 tại L/4, 3L/4) trở thành cơ cấu nên không đạt P đóng;
    nghiệm tối ưu thật có λ1 = λ2 (Olhoff–Rasmussen). dλ_k/dA_e = n·E·α·A_e^(n-1)·φ_kᵀ k̂_e φ_k / (φ_kᵀ G φ_k)."""
    from scipy.optimize import minimize
    n, alpha = fam.n, fam.alpha
    nd = 2 * (N + 1); Le = L / N
    ks = _beam_element_stiffness(1.0, Le); gs = _geo_stiffness(Le)
    G = np.zeros((nd, nd))
    dofs = [[2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3] for e in range(N)]
    for d in dofs:
        G[np.ix_(d, d)] += gs
    fixed = {"pinned": [0, 2 * N], "cantilever": [0, 1], "clamped": [0, 1, 2 * N, 2 * N + 1]}[bc]
    free = [d for d in range(nd) if d not in fixed]
    Gf = G[np.ix_(free, free)]

    def eig(A):
        K = np.zeros((nd, nd)); EI = E * alpha * np.power(A, n)
        for e, d in enumerate(dofs):
            K[np.ix_(d, d)] += EI[e] * ks
        w, Vv = eigh(K[np.ix_(free, free)], Gf, subset_by_index=[0, n_modes - 1])
        return w, Vv

    A0 = np.full(N, V / L) if A_init is None else np.asarray(A_init, float).copy()
    A0 *= V / float(np.sum(A0) * Le)
    w0 = eig(np.full(N, V / L))[0][0]
    x0 = np.concatenate([np.log(A0), [eig(A0)[0][0] / w0]])

    def cons(x):
        return eig(np.exp(x[:N]))[0] / w0 - x[N]

    def jac(x):
        A = np.exp(x[:N]); w, Vv = eig(A)
        J = np.zeros((n_modes, N + 1))
        for k in range(n_modes):
            ph = np.zeros(nd); ph[free] = Vv[:, k]
            den = float(ph @ G @ ph)
            for e, d in enumerate(dofs):
                pe = ph[d]
                J[k, e] = E * alpha * n * A[e] ** (n - 1.0) * float(pe @ ks @ pe) / den * A[e] / w0
            J[k, N] = -1.0
        return J

    lo = math.log(1e-3 * V / L); hi = math.log(50.0 * V / L)
    res = minimize(lambda x: -x[N], x0, jac=lambda x: np.r_[np.zeros(N), -1.0], method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": jac},
                                {"type": "eq", "fun": lambda x: float(np.sum(np.exp(x[:N])) * Le - V),
                                 "jac": lambda x: np.r_[np.exp(x[:N]) * Le, 0.0]}],
                   bounds=[(lo, hi)] * N + [(0.0, 10.0)], options=dict(maxiter=300, ftol=1e-12))
    A = np.exp(res.x[:N]); A *= V / float(np.sum(A) * Le)
    xm = 0.5 * (np.linspace(0, L, N + 1)[:-1] + np.linspace(0, L, N + 1)[1:])
    return xm, A


@dataclass
class ColumnResult:
    bc: str
    L: float
    V: float
    fam: SectionFamily
    E: float
    x: np.ndarray
    A_opt: np.ndarray            # nghiệm E–L dạng đóng (sau khi áp cận dưới nếu có)
    A_uniform: float
    A_numeric: np.ndarray | None
    x_numeric: np.ndarray | None
    P_closed: float              # công thức đóng (không có cận dưới)
    P_fem_opt: float             # FEM cho profile đã dùng
    P_uniform: float
    P_numeric: float | None
    ratio_closed: float
    ratio_fem: float
    floor_frac: float
    history: list[float]
    report: str = ""
    bimodal: bool = False        # True: profile là nghiệm tối ưu đa trị riêng (cột hai đầu ngàm)


def optimize_column(L: float, fam: SectionFamily, E: float, bc: str,
                    V: float | None = None, P_req: float | None = None,
                    floor_frac: float = 0.0, N_check: int = 400,
                    run_numeric: bool = True, N_num: int = 80, iters_num: int = 250) -> ColumnResult:
    """Tối ưu cột. Cho V → tối đa P_cr; hoặc cho P_req → tối thiểu V (cùng nghiệm hình dạng)."""
    n, alpha = fam.n, fam.alpha
    if (V is None) == (P_req is None):
        raise ValueError("Chỉ chọn một trong V hoặc P_req.")
    if P_req is not None:
        # P ∝ V^n  →  V = V1 · (P_req/P1)^(1/n)
        V1 = L * 1e-3
        P1 = column_critical_load_closed(n, alpha, E, L, V1, bc)
        V = V1 * (P_req / P1) ** (1.0 / n)
    B = beta_fn((n + 1.0) / 2.0, 0.5)

    x = np.linspace(0.0, L, 1201)
    A_cf = column_profile(n, bc, L, V, x)

    P_closed = column_critical_load_closed(n, alpha, E, L, V, bc)
    P_uni = column_uniform_load(n, alpha, E, L, V, bc)

    # FEM kiểm chứng trên chính profile đóng (lấy giá trị tại trung điểm phần tử)
    xe = np.linspace(0.0, L, N_check + 1)
    xm = 0.5 * (xe[:-1] + xe[1:])
    Aprof = column_profile(n, bc, L, V, xm)
    Amax = V / (L * (n + 1.0) / (n + 2.0))
    A_floor = max(floor_frac, 1e-4) * Amax
    Aprof = np.maximum(Aprof, A_floor)
    Aprof *= V / float(np.sum(Aprof * np.diff(xe)))
    P_fem, _ = column_buckling_fem(np.diff(xe), E * fam.I(Aprof), bc)

    bimodal = False
    if bc == "clamped":
        # Hai đầu ngàm: profile đóng (đơn mode) có A=0 tại L/4, 3L/4 ⇒ thành cơ cấu, P_fem thực < P đóng
        # (có thể còn thấp hơn cột đều). Nghiệm đúng là bimodal ⇒ dùng tối ưu đa trị riêng.
        N_bi = 60
        xb, Ab = column_bimodal_slsqp(L, V, fam, E, bc, N=N_bi)
        Ab = np.maximum(Ab, A_floor)
        Ab *= V / float(np.sum(Ab) * (L / N_bi))
        Apb = Ab[np.minimum((xm / (L / N_bi)).astype(int), N_bi - 1)]
        P_b, _ = column_buckling_fem(np.diff(xe), E * fam.I(Apb), bc)
        if P_b > P_fem:
            Aprof, P_fem, bimodal = Apb, P_b, True
        if P_req is not None:        # P ∝ V^n chính xác ⇒ hiệu chỉnh V để FEM đạt đúng P_req (không dùng P đóng)
            V_fix = V * (P_req / P_fem) ** (1.0 / n)
            return optimize_column(L, fam, E, bc, V=V_fix, floor_frac=floor_frac, N_check=N_check,
                                   run_numeric=run_numeric, N_num=N_num, iters_num=iters_num)

    # profile hiển thị = profile đã dùng cho FEM (nội suy theo x)
    A_disp = np.interp(x, xm, Aprof)

    A_num = x_num = None
    P_num = None
    hist: list[float] = []
    if run_numeric:
        x_num, A_num, P_num, hist = column_oc_numeric(L, V, fam, E, bc, N=N_num, iters=iters_num)

    res = ColumnResult(bc=bc, L=L, V=V, fam=fam, E=E, x=x, A_opt=A_disp, A_uniform=V / L,
                       A_numeric=A_num, x_numeric=x_num, P_closed=P_closed, P_fem_opt=P_fem,
                       P_uniform=P_uni, P_numeric=P_num, ratio_closed=P_closed / P_uni,
                       ratio_fem=P_fem / P_uni, floor_frac=floor_frac, history=hist, bimodal=bimodal)
    res.report = _column_report(res, B)
    return res


def column_oc_numeric(L: float, V: float, fam: SectionFamily, E: float, bc: str,
                      N: int = 80, iters: int = 250, eta: float = 0.5, floor: float = 1e-3):
    """Tối ưu số (OC): cập nhật A_e theo điều kiện dừng n·A^(n-1)·κ² = hằng, giữ nguyên thể tích."""
    n = fam.n
    xe = np.linspace(0.0, L, N + 1)
    Le = np.diff(xe)
    A = np.full(N, V / L)
    hist: list[float] = []          # P tốt nhất đã đạt sau mỗi vòng (đơn điệu tăng → đồ thị hội tụ rõ ràng)
    best_P, best_A, best_U = -1.0, A.copy(), None
    for _ in range(iters):
        P, U = column_buckling_fem(Le, E * fam.I(A), bc)
        if P > best_P:
            gain = P - best_P
            best_P, best_A, best_U = P, A.copy(), U
            hist.append(best_P)
            if best_P > 0 and gain < 1e-10 * best_P:
                break
        else:                        # bước quá lớn: quay về thiết kế tốt nhất và giảm nửa bước (backtracking)
            eta *= 0.5
            hist.append(best_P)
            if eta < 1e-3:
                break
        kap = np.diff(best_U[1::2]) / Le
        r = best_A ** (n - 1.0) * kap ** 2 + 1e-300
        rbar = float(np.sum(r * best_A * Le) / np.sum(best_A * Le))
        A = best_A * (r / rbar) ** (eta / (n + 1.0))
        A = np.maximum(A, floor * V / L)
        A *= V / float(np.sum(A * Le))
    best = (best_P, best_A)
    P_best, A_best = best
    xm = 0.5 * (xe[:-1] + xe[1:])
    return xm, A_best, P_best, hist


def _column_report(r: ColumnResult, B: float) -> str:
    n = r.fam.n
    L = [
        "=" * 56,
        "   THUYẾT MINH — TỐI ƯU CỘT BẰNG PHÉP TÍNH BIẾN PHÂN",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Sơ đồ: {COLUMN_BCS[r.bc]},  L = {r.L:.3f} m,  E = {r.E:.4g} kN/m²",
        f"   Họ tiết diện: {r.fam.name}   (I = α·A^n, n = {n:g}, α = {r.fam.alpha:.5g})",
        f"   Thể tích (V = ∫A dx) = {r.V:.5g} m³",
        "   Mục tiêu: cực đại tải tới hạn P_cr  (tương đương cực tiểu V với P_cr cho trước).",
        "",
        "II. TRÌNH TỰ BIẾN PHÂN BẬC NHẤT VÀ EULER–LAGRANGE",
        "1) Phiếm hàm Rayleigh với ràng buộc thể tích:",
        "   P[A,w] = ∫E·α·A^n·w''² dx / ∫w'² dx;  ∫A dx = V.",
        "   Lập Lagrangian R = ∫E·α·A^n·w''² dx − P∫w'² dx + λ(∫A dx − V).",
        "2) Biến phân theo A: δR_A = ∫(nEαA^(n−1)w''² + λ)δA dx = 0.",
        "   Do δA tùy ý, bổ đề cơ bản cho nEαA^(n−1)w''² + λ = 0; đổi dấu λ cho quy ước dương:",
        "   nEαA^(n−1)w''² = λ. Vì biểu thức không chứa A', E–L suy biến thành phương trình đại số.",
        "3) Biến phân theo w: tích phân từng phần hai lần cho phương trình trục P(EI w'')'' = 0 ở đoạn không tải ngang.",
        "   Điều kiện biên cơ bản: w đã cho tại gối/ngàm; điều kiện tự nhiên phát sinh từ số hạng biên (mômen/lực cắt).",
        "4) Kết hợp mômen M = EI·w'' = −P·w với điều kiện dừng theo A:",
        "   A ∝ |w|^(2/(n+1)); w'' = −κ·|w|^((1−n)/(n+1))·sign(w).",
        "5) Đổi biến u = A/Amax và tích phân ODE theo điều kiện biên:",
        "   ξ = I_u((n+1)/2, 1/2)  ⇒  u(ξ) = betaincinv((n+1)/2, 1/2, ξ).",
        f"   Trung bình A/Amax = (n+1)/(n+2) = {(n + 1) / (n + 2):.5f}",
        f"   Amax = V / (L·(n+1)/(n+2)) = {r.V / (r.L * (n + 1) / (n + 2)):.5g} m²",
        "",
        "III. THẾ SỐ VÀ ĐỐI CHIẾU",
        f"   B((n+1)/2, 1/2) = {B:.6f}",
        f"   P_cr  (nghiệm đóng)        = {r.P_closed:.5g} kN",
        f"   P_cr  (FEM, profile tối ưu)= {r.P_fem_opt:.5g} kN"
        + (f"   [cận dưới A ≥ {r.floor_frac:.1%}·Amax]" if r.floor_frac > 0 else ""),
        f"   P_cr  (cột đều, cùng V)    = {r.P_uniform:.5g} kN",
        f"   Hệ số tăng tải: đóng = {r.ratio_closed:.4f} ;  FEM = {r.ratio_fem:.4f}",
        "   Trình tự kiểm tra: giữ cùng V, so sánh tải Euler/FEM của cột đều với profile A(x) tối ưu;",
        "   khi có A_min > 0, tải FEM là nghiệm số của profile đã cắt cận, không phải nghiệm đóng không cận.",
    ]
    if r.bimodal:
        L.append("   Cột hai đầu ngàm: nghiệm đơn-mode (A=0 tại L/4, 3L/4) là cơ cấu ⇒ dùng nghiệm BIMODAL (λ₁=λ₂) bằng SLSQP;")
        L.append("   hệ số 'đóng' chỉ là cận tham khảo, không đạt được bằng profile đóng.")
    if r.P_numeric is not None:
        L.append(f"   Tối ưu số OC (FEM)         = {r.P_numeric:.5g} kN  (tỷ số {r.P_numeric / r.P_uniform:.4f})")
    L += [
        "",
        "IV. LƯU Ý",
        "   - Nghiệm E–L cho A → 0 tại các điểm mômen = 0 (hai đầu khớp, đầu tự do, 1/4–3/4 L khi hai đầu ngàm).",
        "     Thực tế phải áp cận dưới A_min (cắt, ổn định cục bộ, liên kết) → P_cr giảm nhẹ.",
        "   - Mô hình Euler–Bernoulli, vật liệu đàn hồi, chưa xét ổn định cục bộ/ứng suất chảy.",
    ]
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════
#  3. DẦM — EULER–LAGRANGE ĐẠI SỐ (dầm tĩnh định) + FEM-OC (mọi sơ đồ)
# ══════════════════════════════════════════════════════════════

BEAM_BCS = {
    "ss": "Dầm tựa đơn",
    "cantilever": "Dầm console (ngàm tại x=0)",
    "fixed_pinned": "Ngàm (x=0) – gối di động (x=L)",
    "fixed_fixed": "Hai đầu ngàm",
}


@dataclass
class BeamProblem:
    L: float
    bc: str
    udls: list[tuple[float, float, float]]          # (q hướng xuống +, x1, x2)
    point_loads: list[tuple[float, float]]          # (P hướng xuống +, x)
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    delta_max: float | None = None                  # m
    x_ctrl: float | None = None
    A_min: float = 1e-4
    A_max: float = 1.0
    n_elem: int = 100


def beam_statics(prob: BeamProblem, x: np.ndarray) -> np.ndarray:
    """Mômen uốn M(x) (căng thớ dưới > 0) cho dầm TĨNH ĐỊNH (ss hoặc cantilever)."""
    L = prob.L
    x = np.asarray(x, float)
    M = np.zeros_like(x)
    if prob.bc == "ss":
        RA = sum(P * (L - xp) / L for P, xp in prob.point_loads)
        RA += sum(q * (b - a) * (L - 0.5 * (a + b)) / L for q, a, b in prob.udls)
        M += RA * x
        for P, xp in prob.point_loads:
            M -= P * np.maximum(x - xp, 0.0)
        for q, a, b in prob.udls:
            inside = (x > a) & (x <= b)
            after = x > b
            M[inside] -= 0.5 * q * (x[inside] - a) ** 2
            M[after] -= q * (b - a) * (x[after] - 0.5 * (a + b))
        return M
    if prob.bc == "cantilever":
        for P, xp in prob.point_loads:
            M -= P * np.maximum(xp - x, 0.0)
        for q, a, b in prob.udls:
            lo = np.maximum(a, x)
            ln = np.maximum(b - lo, 0.0)
            M -= q * ln * (0.5 * (lo + b) - x)
        return M
    raise ValueError("Mômen giải tích chỉ có cho dầm tĩnh định.")


def beam_unit_moment(prob: BeamProblem, x: np.ndarray, xp: float) -> np.ndarray:
    """Mômen m̄(x) do lực đơn vị hướng xuống tại xp (cùng quy ước dấu với beam_statics)."""
    L = prob.L
    x = np.asarray(x, float)
    if prob.bc == "ss":
        return np.where(x <= xp, x * (L - xp) / L, xp * (L - x) / L)
    if prob.bc == "cantilever":
        return -np.maximum(xp - x, 0.0)
    raise ValueError


def beam_closed_form(prob: BeamProblem, m: int = 4001):
    """Nghiệm Euler–Lagrange ĐẠI SỐ:  A(x) = max(A_bền, A_min, c·g^(1/(n+1))),  g = max(M·m̄, 0).
    c xác định từ ràng buộc δ(xp) = δ_max (brentq)."""
    if prob.bc not in ("ss", "cantilever"):
        return None
    fam, E, L = prob.fam, prob.E, prob.L
    n, alpha = fam.n, fam.alpha
    x = np.linspace(0.0, L, m)
    M = beam_statics(prob, x)
    xp = prob.x_ctrl if prob.x_ctrl is not None else (L / 2 if prob.bc == "ss" else L)
    mb = beam_unit_moment(prob, x, xp)
    A_lb = np.maximum(fam.A_from_W(np.abs(M) / prob.sigma_a), prob.A_min)
    g = np.maximum(M * mb, 0.0)

    def A_of(c):
        return np.minimum(np.maximum(A_lb, c * g ** (1.0 / (n + 1.0))), prob.A_max)

    def delta(c):
        A = A_of(c)
        return float(np.trapezoid(M * mb / (E * fam.I(A)), x))

    if prob.delta_max is None or np.max(g) <= 0:
        return dict(x=x, A=A_lb, M=M, c=0.0, delta=delta(0.0), xp=xp, A_lb=A_lb)
    if delta(0.0) <= prob.delta_max:
        return dict(x=x, A=A_lb, M=M, c=0.0, delta=delta(0.0), xp=xp, A_lb=A_lb)
    lo, hi = 1e-6, 1e6
    f = lambda lc: delta(math.exp(lc)) - prob.delta_max
    try:
        lc = brentq(f, math.log(lo), math.log(hi), xtol=1e-12, maxiter=200)
    except ValueError:
        lc = math.log(hi)
    c = math.exp(lc)
    return dict(x=x, A=A_of(c), M=M, c=c, delta=delta(c), xp=xp, A_lb=A_lb)


# ══════════════════════════════════════════════════════════════
#  4. FEM TÁI DÙNG fem_core  (khung phẳng & dàn)
# ══════════════════════════════════════════════════════════════

class FrameFEM:
    """Khung phẳng 2D: nhúng ma trận phần tử / ma trận xoay / tải tương đương / nội lực của fem_core."""
    kind = "frame"
    dof_per_node = 3

    def __init__(self, xy, conn, fam: SectionFamily, E: float, supports, loads, q_local=None):
        self.xy = np.asarray(xy, float)
        self.conn = np.asarray(conn, int)
        self.fam = fam
        self.n_nodes = len(self.xy)
        self.n_el = len(self.conn)
        self.ndof = 3 * self.n_nodes
        self.E = float(E)
        self.q = np.zeros(self.n_el) if q_local is None else np.asarray(q_local, float)

        xi = self.xy[self.conn[:, 0]]; xj = self.xy[self.conn[:, 1]]
        d = xj - xi
        self.L = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.L < 1e-12):
            raise ValueError("Có phần tử chiều dài = 0.")
        self.c = d[:, 0] / self.L; self.s = d[:, 1] / self.L
        self.T = [_rotation_matrix(self.c[e], self.s[e]) for e in range(self.n_el)]
        self.dofs = []
        for e in range(self.n_el):
            i, j = self.conn[e]
            self.dofs.append([3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2])
        self.k_ax = [_frame_element_stiffness_local(self.E, 1.0, 0.0, self.L[e]) for e in range(self.n_el)]
        self.k_bd = [_frame_element_stiffness_local(self.E, 0.0, 1.0, self.L[e]) for e in range(self.n_el)]

        self.F = np.zeros(self.ndof)
        for node, fx, fy, mz in loads:
            self.F[3 * node] += fx; self.F[3 * node + 1] += fy; self.F[3 * node + 2] += mz
        for e in range(self.n_el):
            if self.q[e] != 0.0:
                el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, 1.0, 1.0, float(self.q[e]))
                self.F[self.dofs[e]] += _frame_udl_local_to_global(el, self.L[e], self.T[e])

        con = set()
        self.supports = list(supports)
        for node, ux, uy, rz in self.supports:
            if ux: con.add(3 * node)
            if uy: con.add(3 * node + 1)
            if rz: con.add(3 * node + 2)
        self.constrained = np.array(sorted(con), int)
        self.free = np.array([k for k in range(self.ndof) if k not in con], int)
        self._lu = None
        self._K = None

    # ---- giải
    def solve(self, A):
        A = np.asarray(A, float)
        I = self.fam.I(A)
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            kl = _frame_element_stiffness_local(self.E, A[e], I[e], self.L[e])
            kg = self.T[e].T @ kl @ self.T[e]
            d = self.dofs[e]
            K[np.ix_(d, d)] += kg
        Kff = K[np.ix_(self.free, self.free)]
        lu = lu_factor(Kff)
        U = np.zeros(self.ndof)
        U[self.free] = lu_solve(lu, self.F[self.free])
        self._lu, self._K = lu, K
        return U

    def unit_solve(self, dof: int):
        """Chuyển vị do tải đơn vị tại dof (dùng sau solve cùng A)."""
        rhs = np.zeros(self.ndof); rhs[dof] = 1.0
        Uu = np.zeros(self.ndof)
        Uu[self.free] = lu_solve(self._lu, rhs[self.free])
        return Uu

    def disp_sens(self, U, Uu, A):
        """q_e = Uuᵀ·(∂K/∂A_e)·U  (⇒ ∂δ/∂A_e = -q_e)."""
        A = np.asarray(A, float)
        dI = self.fam.dI_dA(A)
        out = np.zeros(self.n_el)
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]; uu = self.T[e] @ Uu[self.dofs[e]]
            out[e] = uu @ (self.k_ax[e] + dI[e] * self.k_bd[e]) @ ul
        return out

    def forces(self, U, A, stations=(0.0, 0.25, 0.5, 0.75, 1.0)):
        A = np.asarray(A, float)
        I = self.fam.I(A)
        N = np.zeros(self.n_el); M = np.zeros(self.n_el)
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]
            el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, float(A[e]), float(I[e]),
                              float(self.q[e]))
            vals = [_frame_internal_force_at(el, ul, self.L[e], xi * self.L[e]) for xi in stations]
            N[e] = vals[0][0]
            M[e] = max(abs(v[2]) for v in vals)
        return N, M

    def element_diagram(self, U, A, npts=21):
        """(x_global, y_global, N, V, M) tại npts điểm dọc từng phần tử — để vẽ biểu đồ."""
        A = np.asarray(A, float); I = self.fam.I(A)
        out = []
        for e in range(self.n_el):
            ul = self.T[e] @ U[self.dofs[e]]
            el = FrameElement(int(self.conn[e, 0]), int(self.conn[e, 1]), self.E, float(A[e]), float(I[e]),
                              float(self.q[e]))
            xi = np.linspace(0, 1, npts)
            vals = np.array([_frame_internal_force_at(el, ul, self.L[e], t * self.L[e]) for t in xi])
            p0 = self.xy[self.conn[e, 0]]; p1 = self.xy[self.conn[e, 1]]
            out.append(dict(x=p0[0] + xi * (p1[0] - p0[0]), y=p0[1] + xi * (p1[1] - p0[1]),
                            N=vals[:, 0], V=vals[:, 1], M=vals[:, 2]))
        return out

    def reactions(self, U):
        R = self._K @ U - self.F          # K·U = F + R_gối  ⇒  R_gối = K·U − F  (đúng dấu: cân bằng ΣR + ΣF = 0)
        return {int(d): float(R[d]) for d in self.constrained}

    def node_disp(self, U):
        return U.reshape(self.n_nodes, 3)

    def check_stability(self, A0=1e-2):
        K = np.zeros((self.ndof, self.ndof))
        I = self.fam.I(np.full(self.n_el, A0))
        for e in range(self.n_el):
            kl = _frame_element_stiffness_local(self.E, A0, I[e], self.L[e])
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.T[e].T @ kl @ self.T[e]
        w = np.linalg.eigvalsh(K[np.ix_(self.free, self.free)])
        # Cơ cấu thật cho λ_min/λ_max ~ 1e-16…1e-17; lưới mịn (>200 phần tử) cho dầm ổn định có thể xuống ~1e-10,
        # nên ngưỡng 1e-9 cũ báo nhầm "cơ cấu" ⇒ dùng 1e-13.
        if w[0] <= 1e-13 * w[-1]:
            raise ValueError("Kết cấu bất ổn định (cơ cấu) — kiểm tra lại liên kết/gối tựa.")


class TrussFEM:
    """Dàn phẳng liên kết khớp: 2 bậc tự do/nút, mỗi thanh chỉ chịu lực dọc."""
    kind = "truss"
    dof_per_node = 2

    def __init__(self, xy, conn, fam: SectionFamily, E: float, supports, loads, q_local=None):
        self.xy = np.asarray(xy, float)
        self.conn = np.asarray(conn, int)
        self.fam = fam
        self.n_nodes = len(self.xy)
        self.n_el = len(self.conn)
        self.ndof = 2 * self.n_nodes
        self.E = float(E)
        xi = self.xy[self.conn[:, 0]]; xj = self.xy[self.conn[:, 1]]
        d = xj - xi
        self.L = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.L < 1e-12):
            raise ValueError("Có thanh chiều dài = 0.")
        self.c = d[:, 0] / self.L; self.s = d[:, 1] / self.L
        self.b = np.stack([-self.c, -self.s, self.c, self.s], axis=1)
        self.dofs = [[2 * i, 2 * i + 1, 2 * j, 2 * j + 1] for i, j in self.conn]
        self.F = np.zeros(self.ndof)
        for node, fx, fy, mz in loads:
            self.F[2 * node] += fx; self.F[2 * node + 1] += fy
        con = set()
        self.supports = list(supports)
        for node, ux, uy, rz in self.supports:
            if ux: con.add(2 * node)
            if uy: con.add(2 * node + 1)
        self.constrained = np.array(sorted(con), int)
        self.free = np.array([k for k in range(self.ndof) if k not in con], int)
        self.q = np.zeros(self.n_el)
        self._lu = None
        self._K = None

    def solve(self, A):
        A = np.asarray(A, float)
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            b = self.b[e]
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.E * A[e] / self.L[e] * np.outer(b, b)
        lu = lu_factor(K[np.ix_(self.free, self.free)])
        U = np.zeros(self.ndof)
        U[self.free] = lu_solve(lu, self.F[self.free])
        self._lu, self._K = lu, K
        return U

    def unit_solve(self, dof: int):
        rhs = np.zeros(self.ndof); rhs[dof] = 1.0
        Uu = np.zeros(self.ndof)
        Uu[self.free] = lu_solve(self._lu, rhs[self.free])
        return Uu

    def disp_sens(self, U, Uu, A):
        out = np.zeros(self.n_el)
        for e in range(self.n_el):
            d = self.dofs[e]
            out[e] = self.E / self.L[e] * float(self.b[e] @ Uu[d]) * float(self.b[e] @ U[d])
        return out

    def forces(self, U, A, stations=None):
        A = np.asarray(A, float)
        N = np.array([self.E * A[e] / self.L[e] * float(self.b[e] @ U[self.dofs[e]]) for e in range(self.n_el)])
        return N, np.zeros(self.n_el)

    def reactions(self, U):
        R = self._K @ U - self.F          # K·U = F + R_gối  ⇒  R_gối = K·U − F  (đúng dấu: cân bằng ΣR + ΣF = 0)
        return {int(d): float(R[d]) for d in self.constrained}

    def node_disp(self, U):
        return U.reshape(self.n_nodes, 2)

    def check_stability(self, A0=1e-2):
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n_el):
            K[np.ix_(self.dofs[e], self.dofs[e])] += self.E * A0 / self.L[e] * np.outer(self.b[e], self.b[e])
        w = np.linalg.eigvalsh(K[np.ix_(self.free, self.free)])
        if w[0] <= 1e-9 * w[-1]:
            raise ValueError("Dàn bất ổn định (cơ cấu) — thiếu thanh hoặc gối tựa.")


# ══════════════════════════════════════════════════════════════
#  5. BỘ TỐI ƯU: FSD (bền/ổn định) + OC (chuyển vị)
# ══════════════════════════════════════════════════════════════
#  min  W(A) = ρ·Σ A_e·L_e
#  s.t. U_e(A) = |N|/(A·σ_N) + |M|/(W(A)·σ_a) ≤ 1     (bền + ổn định Euler khi nén)
#       |δ_c(A)| ≤ δ_c,max
#  Điều kiện tối ưu (Euler–Lagrange rời rạc): B_e = -(∂δ/∂A_e)/L_e = Λ  đều cho các phần tử
#  không chạm cận (cận bền / A_min / A_max).

@dataclass
class OptSettings:
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    rho: float = STEEL_RHO
    FS_buckling: float = 1.0
    check_buckling: bool = True
    A_min: float = 1e-4
    A_max: float = 1.0
    disp_limits: list = field(default_factory=list)      # [(dof_toàn_cục, giới hạn>0)]
    L_buck: np.ndarray | None = None
    eta: float = 0.5
    relax: float = 0.7
    max_iter: int = 200
    tol: float = 1e-6
    p_active: float = 4.0
    multistart: int = 10            # số điểm khởi tạo ngẫu nhiên tìm cực tiểu cục bộ khác (kết cấu nhỏ, bài toán không lồi)
    ms_max_elems: int = 24          # chỉ chạy multistart khi số phần tử ≤ ngưỡng này
    ms_time_s: float = 8.0          # ngân sách thời gian cho multistart


def utilization(A, N, M, st: OptSettings, Lb):
    A = np.asarray(A, float)
    I = st.fam.I(A); W = st.fam.W(A)
    sig_N = np.full_like(A, st.sigma_a)
    if st.check_buckling:
        sig_cr = (math.pi ** 2 * st.E * I / (A * np.asarray(Lb) ** 2)) / st.FS_buckling
        sig_N = np.where(N < 0, np.minimum(st.sigma_a, sig_cr), st.sigma_a)
    return np.abs(N) / (A * sig_N) + np.abs(M) / (W * st.sigma_a)


def required_area(N, M, st: OptSettings, Lb):
    n_el = len(N)
    lo = np.log(np.full(n_el, st.A_min)); hi = np.log(np.full(n_el, st.A_max))
    ok_lo = utilization(np.exp(lo), N, M, st, Lb) <= 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        feas = utilization(np.exp(mid), N, M, st, Lb) <= 1.0
        hi = np.where(feas, mid, hi)
        lo = np.where(feas, lo, mid)
    A = np.exp(hi)
    return np.where(ok_lo, st.A_min, A)


@dataclass
class OptResult:
    A: np.ndarray
    U: np.ndarray
    N: np.ndarray
    M: np.ndarray
    util: np.ndarray
    disp_ratio: np.ndarray
    weight: float
    weight_uniform: float
    A_uniform: float
    history: list[dict]
    converged: bool
    iters: int
    feasible: bool
    B: np.ndarray | None
    EL_cv: float | None
    L_el: np.ndarray


def _weight(A, model, st):
    return float(st.rho * np.sum(np.asarray(A) * model.L))


def _volume(A, model) -> float:
    return float(np.sum(np.asarray(A) * model.L))


def _disp_grads(model, st, A, U):
    """Giá trị g_k = 1-|δ_k|/lim_k và ∂g_k/∂A_e (liên hợp). model.solve(A) phải vừa được gọi."""
    G = np.zeros((len(st.disp_limits), model.n_el)); g = np.zeros(len(st.disp_limits))
    for k, (dof, lim) in enumerate(st.disp_limits):
        Uu = model.unit_solve(dof)
        qe = model.disp_sens(U, Uu, A)
        sgn = 1.0 if U[dof] >= 0 else -1.0
        g[k] = 1.0 - abs(U[dof]) / lim
        G[k] = sgn * qe / lim
    return g, G


def _kkt_polish(model, st, A, Lb, evaluate, hist):
    """Vòng ngoài: cố định cận bền A_lb (FSD); vòng trong: min W s.t. ràng buộc chuyển vị, A_lb ≤ A ≤ A_max."""
    from scipy.optimize import minimize
    ne = model.n_el
    W0 = max(_weight(A, model, st), 1e-30)
    A_best, W_best = None, np.inf
    for outer in range(20):
        U, N, M, u, r = evaluate(A)
        A_lb = np.clip(required_area(N, M, st, Lb), st.A_min, st.A_max)
        y0 = np.log(np.clip(np.maximum(A, A_lb), st.A_min, st.A_max))
        cache = {}

        def ev(y):
            key = y.tobytes()
            if key not in cache:
                Ay = np.exp(y)
                Uy = model.solve(Ay)
                cache.clear(); cache[key] = (Ay, Uy)
            return cache[key]

        def f(y): return float(st.rho * np.sum(np.exp(y) * model.L) / W0)
        def fg(y): return st.rho * np.exp(y) * model.L / W0
        def cons(y):
            Ay, Uy = ev(y)
            return np.array([1.0 - abs(Uy[d]) / lim for d, lim in st.disp_limits])
        def cjac(y):
            Ay, Uy = ev(y); model.solve(Ay)
            _, G = _disp_grads(model, st, Ay, Uy)
            return G * Ay[None, :]

        sol = minimize(f, y0, jac=fg, method="SLSQP",
                       bounds=list(zip(np.log(A_lb), np.full(ne, math.log(st.A_max)))),
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options=dict(maxiter=500, ftol=1e-14))
        A_new = np.clip(np.exp(sol.x), st.A_min, st.A_max)
        Un, Nn, Mn, un, rn = evaluate(A_new)
        feas = (np.max(un) <= 1.0 + 1e-6) and (rn.size == 0 or np.max(rn) <= 1.0 + 1e-6)
        Wn = _weight(A_new, model, st)
        hist.append(dict(iter=len(hist) + 1, weight=Wn, max_util=float(np.max(un)),
                         max_disp_ratio=float(np.max(rn)) if rn.size else 0.0,
                         dA=float(np.max(np.abs(np.log(A_new / A))))))
        if feas and Wn < W_best:
            A_best, W_best = A_new.copy(), Wn
        dA = float(np.max(np.abs(np.log(A_new / A))))
        A = A_new
        if dA < 1e-4:
            break
    if A_best is None:
        return A, None
    evaluate(A_best)
    return A_best, None


def _multistart_full(model, st, Lb, evaluate, A_best, W_best, hist):
    """Bài toán min W s.t. (bền + ổn định + chuyển vị) KHÔNG LỒI khi kết cấu siêu tĩnh (nội lực phụ thuộc A) ⇒ có nhiều
    cực tiểu cục bộ (vd. dàn 10 thanh Schmit: 5060.85 và 5076.7 lb). FSD/OC + `_kkt_polish` luôn rơi vào một điểm dừng;
    ở đây chạy thêm SLSQP trên TOÀN BỘ ràng buộc (đạo hàm sai phân) từ nhiều điểm khởi tạo ngẫu nhiên và giữ nghiệm nhẹ nhất."""
    import time
    from scipy.optimize import minimize
    ne = model.n_el
    if st.multistart <= 0 or ne > st.ms_max_elems:
        return A_best, W_best
    rng = np.random.default_rng(12345)
    W0 = max(W_best, 1e-30)
    lo, hi = math.log(st.A_min), math.log(st.A_max)

    def cons(y):
        A = np.exp(y); U = model.solve(A); N, M = model.forces(U, A)
        c = list(1.0 - utilization(A, N, M, st, Lb))
        c += [1.0 - abs(U[d]) / lim for d, lim in st.disp_limits]
        return np.array(c)

    t0 = time.monotonic()
    yb = np.log(A_best)
    for k in range(st.multistart):
        if time.monotonic() - t0 > st.ms_time_s:
            break
        if k % 2 == 0:
            y0 = np.log(A_best) + rng.normal(0.0, 0.6, ne)
        else:
            y0 = np.log(np.clip(A_best.mean(), st.A_min, st.A_max)) + rng.normal(0.0, 1.2, ne)
        y0 = np.clip(y0, lo, hi)
        try:
            sol = minimize(lambda y: float(st.rho * np.sum(np.exp(y) * model.L) / W0), y0,
                           jac=lambda y: st.rho * np.exp(y) * model.L / W0, method="SLSQP",
                           bounds=[(lo, hi)] * ne, constraints=[{"type": "ineq", "fun": cons}],
                           options=dict(maxiter=300, ftol=1e-13))
        except Exception:
            continue
        A_c = np.clip(np.exp(sol.x), st.A_min, st.A_max)
        U, N, M, u, r = evaluate(A_c)
        feas = (np.max(u) <= 1.0 + 1e-6) and (r.size == 0 or np.max(r) <= 1.0 + 1e-6)
        Wc = _weight(A_c, model, st)
        if feas and Wc < W_best * (1.0 - 1e-9):
            A_best, W_best = A_c.copy(), Wc
            hist.append(dict(iter=len(hist) + 1, weight=Wc, max_util=float(np.max(u)),
                             max_disp_ratio=float(np.max(r)) if r.size else 0.0, dA=0.0))
    return A_best, W_best


def _kkt_residual(model, st, A, U, Lb, N, M):
    """‖∇W − Σλ_k ∇δ_k‖ / ‖∇W‖ trên các phần tử tự do, λ ≥ 0 giải bằng NNLS (điều kiện Euler–Lagrange rời rạc)."""
    from scipy.optimize import nnls
    model.solve(A)
    _, G = _disp_grads(model, st, A, U)
    A_lb = required_area(N, M, st, Lb)
    free = (A > 1.02 * A_lb) & (A > 1.02 * st.A_min) & (A < 0.98 * st.A_max)
    if np.sum(free) < 2:
        return None
    gradW = st.rho * model.L
    act = np.array([abs(U[d]) / lim > 0.5 for d, lim in st.disp_limits])
    if not act.any():
        return None
    Gm = (G[act][:, free]).T            # KKT: ∇W = Σ λ_k ∇g_k  với g_k = 1-|δ_k|/lim ≥ 0, λ_k ≥ 0
    lam, res = nnls(Gm * A[free][:, None], gradW[free] * A[free])
    return float(res / np.linalg.norm(gradW[free] * A[free]))


def optimize_structure(model, st: OptSettings, A_init: np.ndarray | None = None) -> OptResult:
    ne = model.n_el
    Lb = st.L_buck if st.L_buck is not None else np.ones(ne)
    model.check_stability()

    def evaluate(A):
        U = model.solve(A)
        N, M = model.forces(U, A)
        u = utilization(A, N, M, st, Lb)
        r = np.array([abs(U[d]) / lim for d, lim in st.disp_limits]) if st.disp_limits else np.zeros(0)
        return U, N, M, u, r

    # --- thiết kế đều (một tiết diện cho mọi phần tử) = mốc so sánh
    def g_uniform(la):
        A = np.full(ne, math.exp(la))
        _, _, _, u, r = evaluate(A)
        return max(float(np.max(u)), float(np.max(r)) if r.size else 0.0) - 1.0

    la_lo, la_hi = math.log(st.A_min), math.log(st.A_max)
    if g_uniform(la_lo) <= 0:
        A_u = st.A_min
    elif g_uniform(la_hi) > 0:
        A_u = st.A_max
    else:
        A_u = math.exp(brentq(g_uniform, la_lo, la_hi, xtol=1e-10, maxiter=200))
    W_uniform = _weight(np.full(ne, A_u), model, st)

    A = np.full(ne, A_u) if A_init is None else np.asarray(A_init, float).copy()
    hist = []
    B_last = None
    converged = False
    it = 0
    for it in range(1, st.max_iter + 1):
        U, N, M, u, r = evaluate(A)
        A_lb = required_area(N, M, st, Lb)
        A_new = A_lb.copy()
        if st.disp_limits:
            Bsum = np.zeros(ne)
            for (dof, lim), rc in zip(st.disp_limits, r):
                Uu = model.unit_solve(dof)
                qe = model.disp_sens(U, Uu, A)
                sgn = 1.0 if U[dof] >= 0 else -1.0
                w = max(rc, 1e-3) ** st.p_active
                Bsum += w * np.maximum(sgn * qe, 0.0) / lim
            B = Bsum / model.L
            Bmax = float(np.max(B)) if np.max(B) > 0 else 1.0
            B = np.maximum(B, 1e-9 * Bmax)
            Bbar = float(np.sum(B * A * model.L) / np.sum(A * model.L))
            A_trial = A * (B / Bbar) ** st.eta
            B_last = B

            def Afun(s):
                return np.clip(np.maximum(s * A_trial, A_lb), st.A_min, st.A_max)

            def g(ls):
                Ux = model.solve(Afun(math.exp(ls)))
                return max(abs(Ux[d]) / lim for d, lim in st.disp_limits) - 1.0

            lo_s, hi_s = math.log(1e-3), math.log(1e3)
            if g(lo_s) <= 0:
                s = math.exp(lo_s)
            elif g(hi_s) > 0:
                s = math.exp(hi_s)
            else:
                s = math.exp(brentq(g, lo_s, hi_s, xtol=1e-10, maxiter=100))
            A_new = Afun(s)
            model.solve(A)  # khôi phục factor cho vòng lặp kế (unit_solve dùng cache)
        A_next = np.exp((1 - st.relax) * np.log(A) + st.relax * np.log(A_new))
        Wn = _weight(A_next, model, st)
        dA = float(np.max(np.abs(np.log(A_next / A))))
        hist.append(dict(iter=it, weight=Wn, max_util=float(np.max(u)),
                         max_disp_ratio=float(np.max(r)) if r.size else 0.0, dA=dA))
        A = A_next
        if len(hist) > 2 and dA < 1e-4 and abs(hist[-1]["weight"] - hist[-2]["weight"]) < st.tol * Wn:
            converged = True
            break

    # --- hoàn thiện KKT: nhân tử Lagrange thật cho các ràng buộc chuyển vị (SLSQP + đạo hàm liên hợp)
    EL_res = None
    if st.disp_limits:
        A, EL_res = _kkt_polish(model, st, A, Lb, evaluate, hist)

    # --- tìm thêm cực tiểu cục bộ (multistart, chỉ kết cấu nhỏ)
    if st.multistart > 0 and ne <= st.ms_max_elems:
        U_, N_, M_, u_, r_ = evaluate(A)
        if np.max(u_) <= 1.0 + 1e-6 and (r_.size == 0 or np.max(r_) <= 1.0 + 1e-6):
            A, _w = _multistart_full(model, st, Lb, evaluate, A, _weight(A, model, st), hist)

    # --- hiệu chỉnh cuối để đảm bảo thoả mãn ràng buộc
    for _ in range(80):
        U, N, M, u, r = evaluate(A)
        rmax = float(np.max(r)) if r.size else 0.0
        if np.max(u) <= 1.0 + 1e-9 and rmax <= 1.0 + 1e-9:
            break
        A = np.where(u > 1.0, A * u, A)
        if rmax > 1.0 + 1e-9:
            A = A * rmax ** 0.7
        A = np.clip(A, st.A_min, st.A_max)
    U, N, M, u, r = evaluate(A)
    feasible = bool(np.max(u) <= 1.0 + 1e-6 and (r.size == 0 or np.max(r) <= 1.0 + 1e-6))

    # --- sai số điều kiện E–L rời rạc (KKT) trên các phần tử "tự do" (không chạm cận)
    EL_cv = None
    if st.disp_limits:
        EL_cv = _kkt_residual(model, st, A, U, Lb, N, M)
    return OptResult(A=A, U=U, N=N, M=M, util=u, disp_ratio=r, weight=_weight(A, model, st),
                     weight_uniform=W_uniform, A_uniform=A_u, history=hist, converged=converged, iters=it,
                     feasible=feasible, B=B_last, EL_cv=EL_cv, L_el=model.L.copy())


# ══════════════════════════════════════════════════════════════
#  6. DẦM — ĐƯA BÀI TOÁN DẦM VỀ FEM + SO SÁNH NGHIỆM ĐÓNG
# ══════════════════════════════════════════════════════════════

@dataclass
class BeamOptResult:
    prob: BeamProblem
    xs: np.ndarray                    # toạ độ nút
    x_mid: np.ndarray
    A_opt: np.ndarray
    A_uniform: float
    I_opt: np.ndarray
    opt: OptResult
    closed: dict | None
    deflection_opt: np.ndarray        # võng tại nút (m, hướng xuống +)
    deflection_uni: np.ndarray
    moment_x: np.ndarray
    moment_opt: np.ndarray
    vol_opt: float
    vol_uniform: float
    vol_closed: float | None
    delta_ctrl_opt: float | None
    delta_ctrl_uni: float | None
    ctrl_node: int
    crosscheck: dict
    report: str = ""


def _beam_mesh(prob: BeamProblem):
    L = prob.L
    pts = list(np.linspace(0.0, L, prob.n_elem + 1))
    pts += [x for _, x in prob.point_loads]
    for _, a, b in prob.udls:
        pts += [a, b]
    xp = prob.x_ctrl if prob.x_ctrl is not None else (L / 2 if prob.bc != "cantilever" else L)
    pts.append(xp)
    pts = np.array(sorted(pts))
    keep = [pts[0]]
    for p in pts[1:]:
        if p - keep[-1] > 1e-9 * L:
            keep.append(p)
    return np.array(keep), xp


def build_beam_model(prob: BeamProblem):
    xs, xp = _beam_mesh(prob)
    nn = len(xs); ne = nn - 1
    xy = np.stack([xs, np.zeros(nn)], axis=1)
    conn = np.stack([np.arange(ne), np.arange(1, ne + 1)], axis=1)
    sup = {"ss": [(0, True, True, False), (nn - 1, False, True, False)],
           "cantilever": [(0, True, True, True)],
           "fixed_pinned": [(0, True, True, True), (nn - 1, False, True, False)],
           "fixed_fixed": [(0, True, True, True), (nn - 1, False, True, True)]}[prob.bc]
    loads = []
    for P, x in prob.point_loads:
        k = int(np.argmin(np.abs(xs - x)))
        loads.append((k, 0.0, -P, 0.0))
    q_local = np.zeros(ne)
    xm = 0.5 * (xs[:-1] + xs[1:])
    for q, a, b in prob.udls:
        q_local[(xm > a) & (xm < b)] += -q          # hướng xuống ⇒ trục y cục bộ âm
    model = FrameFEM(xy, conn, prob.fam, prob.E, sup, loads, q_local)
    model._nodal_loads = list(loads)
    ctrl = int(np.argmin(np.abs(xs - xp)))
    return model, xs, ctrl


def optimize_beam(prob: BeamProblem) -> BeamOptResult:
    model, xs, ctrl = build_beam_model(prob)
    ne = model.n_el
    limits = []
    if prob.delta_max is not None:
        limits.append((3 * ctrl + 1, prob.delta_max))
    st = OptSettings(fam=prob.fam, E=prob.E, sigma_a=prob.sigma_a, check_buckling=False,
                     A_min=prob.A_min, A_max=prob.A_max, disp_limits=limits, max_iter=300, relax=0.8)
    opt = optimize_structure(model, st)

    # chuyển vị của thiết kế đều (cùng ràng buộc)
    A_u = np.full(ne, opt.A_uniform)
    U_u = model.solve(A_u)
    U_o = model.solve(opt.A)
    model.solve(opt.A)

    # mômen theo trục x (tại nút) từ phần tử: lấy M tại hai đầu mỗi phần tử
    diag = model.element_diagram(U_o, opt.A, npts=3)
    mx = []; my = []
    for e, d in enumerate(diag):
        mx += [d["x"][0], d["x"][1]]
        my += [d["M"][0], d["M"][1]]
    mx.append(diag[-1]["x"][-1]); my.append(diag[-1]["M"][-1])

    closed = beam_closed_form(prob)
    x_mid = 0.5 * (xs[:-1] + xs[1:])
    vol_o = float(np.sum(opt.A * model.L))
    vol_u = float(opt.A_uniform * prob.L)
    vol_c = float(np.trapezoid(closed["A"], closed["x"])) if closed is not None else None
    d_o = d_u = None
    if limits:
        d_o = abs(float(U_o[3 * ctrl + 1])); d_u = abs(float(U_u[3 * ctrl + 1]))

    cross = crosscheck_with_core(model, opt.A)
    res = BeamOptResult(prob=prob, xs=xs, x_mid=x_mid, A_opt=opt.A, A_uniform=opt.A_uniform,
                        I_opt=prob.fam.I(opt.A), opt=opt, closed=closed,
                        deflection_opt=-U_o[1::3], deflection_uni=-U_u[1::3],
                        moment_x=np.array(mx), moment_opt=np.array(my),
                        vol_opt=vol_o, vol_uniform=vol_u, vol_closed=vol_c,
                        delta_ctrl_opt=d_o, delta_ctrl_uni=d_u, ctrl_node=ctrl, crosscheck=cross)
    res.report = _beam_report(res)
    return res


def crosscheck_with_core(model: FrameFEM, A) -> dict:
    """Đối chiếu bộ giải nhanh với fem_core.solve_plane_frame (nguyên bản)."""
    A = np.asarray(A, float)
    I = model.fam.I(A)
    nodes = [FrameNode(float(p[0]), float(p[1])) for p in model.xy]
    elems = [FrameElement(int(model.conn[e, 0]), int(model.conn[e, 1]), model.E, float(A[e]), float(I[e]),
                          float(model.q[e])) for e in range(model.n_el)]
    sups = [FrameSupport(node, bool(ux), bool(uy), bool(rz)) for node, ux, uy, rz in model.supports]
    # tải nút gốc (không gồm tải tương đương của UDL — core tự tính từ udl_local)
    pls = [FramePointLoad(node, fx, fy, mz) for node, fx, fy, mz in getattr(model, "_nodal_loads", [])]
    ref = solve_plane_frame(PlaneFrameInput(nodes, elems, sups, pls))
    U = model.solve(A)
    U_ref = ref.node_displacements.reshape(-1)
    dmax = float(np.max(np.abs(U - U_ref)))
    scale = float(np.max(np.abs(U_ref))) + 1e-300
    return dict(max_abs_diff=dmax, rel_diff=dmax / scale)


# ══════════════════════════════════════════════════════════════
#  7. KHUNG / DÀN — NHẬP LIỆU VÀ TỐI ƯU TOÀN KẾT CẤU
# ══════════════════════════════════════════════════════════════

@dataclass
class StructInput:
    nodes: list[tuple[float, float]]
    members: list[tuple[int, int]]
    supports: list[tuple[int, bool, bool, bool]]          # (node, ux, uy, rz)
    loads: list[tuple[int, float, float, float]]          # (node, Fx, Fy, Mz)
    member_q: list[float]                                 # q vuông góc thanh (khung)
    mode: str                                             # "frame" | "truss"
    fam: SectionFamily
    E: float = STEEL_E
    sigma_a: float = STEEL_SIGMA
    rho: float = STEEL_RHO
    FS_buckling: float = 1.0
    k_eff: float = 1.0
    check_buckling: bool = True
    A_min: float = 1e-4
    A_max: float = 1.0
    disp_limits: list[tuple[int, str, float]] = field(default_factory=list)   # (node, 'ux'|'uy', |δ|max m)
    segments: int = 1
    # Tải NGƯỜI DÙNG nhập (chỉ để vẽ sơ đồ tải; solver dùng `loads` và `member_q` đã quy đổi)
    user_nodal: list[tuple[int, float, float, float]] = field(default_factory=list)   # (node, Fx, Fy, Mz)
    user_udl: list[tuple[int, float]] = field(default_factory=list)                   # (thanh, w kN/m hướng xuống)


@dataclass
class StructOptResult:
    inp: StructInput
    model: object
    member_of_el: np.ndarray
    opt: OptResult
    A_member: np.ndarray            # diện tích trung bình theo thanh gốc
    weight: float
    weight_uniform: float
    saving: float
    crosscheck: dict | None
    reactions: dict
    report: str = ""


def build_structure(inp: StructInput):
    nodes = [tuple(p) for p in inp.nodes]
    conn = []
    member_of_el = []
    q_el = []
    seg = max(1, int(inp.segments)) if inp.mode == "frame" else 1
    for m, (i, j) in enumerate(inp.members):
        if seg == 1:
            conn.append((i, j)); member_of_el.append(m)
            q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
        else:
            pi, pj = np.array(nodes[i]), np.array(nodes[j])
            prev = i
            for k in range(1, seg):
                p = pi + (pj - pi) * k / seg
                nodes.append((float(p[0]), float(p[1])))
                cur = len(nodes) - 1
                conn.append((prev, cur)); member_of_el.append(m)
                q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
                prev = cur
            conn.append((prev, j)); member_of_el.append(m)
            q_el.append(inp.member_q[m] if m < len(inp.member_q) else 0.0)
    cls = FrameFEM if inp.mode == "frame" else TrussFEM
    model = cls(nodes, conn, inp.fam, inp.E, inp.supports, inp.loads, q_el)
    model._nodal_loads = list(inp.loads)
    member_of_el = np.array(member_of_el, int)
    mem_len = np.array([math.hypot(inp.nodes[j][0] - inp.nodes[i][0], inp.nodes[j][1] - inp.nodes[i][1])
                        for i, j in inp.members])
    Lb = inp.k_eff * mem_len[member_of_el]
    dpn = model.dof_per_node
    limits = []
    for node, d, lim in inp.disp_limits:
        limits.append((dpn * node + (0 if d == "ux" else 1), float(lim)))
    st = OptSettings(fam=inp.fam, E=inp.E, sigma_a=inp.sigma_a, rho=inp.rho,
                     FS_buckling=inp.FS_buckling, check_buckling=inp.check_buckling,
                     A_min=inp.A_min, A_max=inp.A_max, disp_limits=limits, L_buck=Lb)
    return model, st, member_of_el


def optimize_frame(inp: StructInput) -> StructOptResult:
    if not inp.nodes or not inp.members:
        raise ValueError("Chưa có nút/thanh.")
    if not inp.supports:
        raise ValueError("Chưa khai báo gối tựa.")
    model, st, member_of_el = build_structure(inp)
    opt = optimize_structure(model, st)
    nm = len(inp.members)
    A_member = np.zeros(nm)
    for m in range(nm):
        sel = member_of_el == m
        A_member[m] = float(np.sum(opt.A[sel] * model.L[sel]) / np.sum(model.L[sel]))
    model.solve(opt.A)
    reac = model.reactions(opt.U)
    cross = None
    if inp.mode == "frame":
        cross = crosscheck_with_core(model, opt.A)
    res = StructOptResult(inp=inp, model=model, member_of_el=member_of_el, opt=opt, A_member=A_member,
                          weight=opt.weight, weight_uniform=opt.weight_uniform,
                          saving=1.0 - opt.weight / opt.weight_uniform if opt.weight_uniform > 0 else 0.0,
                          crosscheck=cross, reactions=reac)
    res.report = _struct_report(res)
    return res


# ══════════════════════════════════════════════════════════════
#  8. BÁO CÁO
# ══════════════════════════════════════════════════════════════

def _beam_report(r: BeamOptResult) -> str:
    p = r.prob; n = p.fam.n
    L = [
        "=" * 56,
        "   THUYẾT MINH — TỐI ƯU DẦM VÚT THON (BIẾN PHÂN + FEM)",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Sơ đồ: {BEAM_BCS[p.bc]},  L = {p.L:.3f} m,  E = {p.E:.4g} kN/m²,  [σ] = {p.sigma_a / 1e3:.1f} MPa",
        f"   Họ tiết diện: {p.fam.name}  (I = α·A^n, n = {n:g};  W = β·A^(n-γ), γ = {p.fam.gamma:g})",
        "   Tải trọng (hướng xuống dương):",
    ]
    for P, x in p.point_loads:
        L.append(f"     - P = {P:.3f} kN tại x = {x:.3f} m")
    for q, a, b in p.udls:
        L.append(f"     - q = {q:.3f} kN/m từ x = {a:.3f} đến {b:.3f} m")
    if p.delta_max is not None:
        xp = r.xs[r.ctrl_node]
        L.append(f"   Ràng buộc: |δ(x={xp:.3f} m)| ≤ {p.delta_max * 1e3:.3f} mm;  A ≥ {p.A_min:.3g} m²")
    else:
        L.append("   Ràng buộc: chỉ điều kiện bền (σ = |M|/W ≤ [σ])")
    L += [
        "",
        "II. THIẾT LẬP ĐỘ VÕNG BẰNG NGUYÊN LÝ CÔNG ẢO",
        "1) Hệ thực: giải cân bằng để có M(x); hệ đơn vị: đặt lực đơn vị tại x_p để có m̄(x).",
        "   δ(x_p) = ∫₀ᴸ M(x)m̄(x)/(E I(x)) dx = ∫₀ᴸ M·m̄/(EαA^n) dx.",
        "2) Bài toán: min V[A] = ∫₀ᴸ A dx, với δ(x_p) ≤ δ_max và cận vật liệu A_min ≤ A ≤ A_max.",
        "3) Với phương trình dầm, δΠ = ∫EIw''η''dx − ∫qηdx; tích phân từng phần hai lần cho:",
        "   δΠ = ∫[(EIw'')'' − q]ηdx + [EIw''η' − (EIw'')'η]₀ᴸ.",
        "   Ở đầu tự do, η và η' độc lập ⇒ điều kiện tự nhiên M = EIw'' = 0 và V = (EIw'')' = 0.",
        "   Ở gối/ngàm, chuyển vị và/hoặc góc xoay bị áp đặt là điều kiện biên cơ bản.",
        "",
        "III. SUY DẪN BIẾN PHÂN BẬC NHẤT / EULER–LAGRANGE",
        "1) Nếu ràng buộc võng hoạt động, lập J[A] = ∫A dx + λ(∫M·m̄/(EαA^n)dx − δ_max).",
        "2) Lấy biến phân A → A + εη; δJ = ∫[1 − λnM·m̄/(EαA^(n+1))]η dx.",
        "   Vì phiếm hàm không chứa A', sau tích phân từng phần số hạng biên bằng 0;",
        "   bổ đề cơ bản cho ngoặc vuông bằng 0 (E–L suy biến thành phương trình đại số):",
        "      1 − λnM·m̄/(EαA^(n+1)) = 0.",
        f"3) Giải theo A: A(x) = [λnM(x)m̄(x)/(Eα)]^(1/(n+1)); với n={n:g}, "
        f"A ∝ |M·m̄|^(1/{n + 1:g}) trên vùng tự do.",
        "4) Áp cận cục bộ: A_bền = A_from_W(|M|/[σ]); A = max(A_bền, A_min, c·g^(1/(n+1))),",
        "   g = max(M·m̄, 0); c được tìm từ phương trình δ(c) = δ_max.",
    ]
    if r.closed is not None:
        L.append(f"   c = {r.closed['c']:.5g}   |   δ(nghiệm đóng) = {r.closed['delta'] * 1e3:.4f} mm")
    else:
        L.append("   Dầm siêu tĩnh: M phụ thuộc A, nên bước đóng A(M,m̄) không giải tách được; chuyển sang FEM + OC/KKT.")
    L += [
        "",
        "IV. KẾT QUẢ VÀ ĐỐI CHIẾU SỐ",
        f"   Thể tích tối ưu (FEM-OC)    = {r.vol_opt:.5g} m³",
    ]
    if r.vol_closed is not None:
        L.append(f"   Thể tích nghiệm đóng (E–L)  = {r.vol_closed:.5g} m³  "
                 f"(lệch {abs(r.vol_opt - r.vol_closed) / r.vol_closed * 100:.2f}%)")
    L += [
        f"   Thể tích dầm đều (1 tiết diện) = {r.vol_uniform:.5g} m³   [A = {r.A_uniform:.5g} m²]",
        f"   Tiết kiệm vật liệu            = {(1 - r.vol_opt / r.vol_uniform) * 100:.2f}%",
        f"   A_max/A_min của thiết kế     = {np.max(r.A_opt):.5g} / {np.min(r.A_opt):.5g} m²",
    ]
    if r.delta_ctrl_opt is not None:
        L.append(f"   δ tại điểm khống chế: tối ưu = {r.delta_ctrl_opt * 1e3:.4f} mm ; dầm đều = "
                 f"{r.delta_ctrl_uni * 1e3:.4f} mm ; giới hạn = {p.delta_max * 1e3:.4f} mm")
    L += [
        f"   Hội tụ: {'có' if r.opt.converged else 'chưa'} sau {r.opt.iters} vòng lặp; "
        f"thoả ràng buộc: {'có' if r.opt.feasible else 'KHÔNG'}",
        "   Kiểm tra số: FEM-OC so với tiết diện đều dưới cùng tải và giới hạn; nghiệm đóng chỉ dùng cho dầm tĩnh định.",
        f"   Đối chiếu bộ giải nhanh với fem_core.solve_plane_frame: sai khác chuyển vị tương đối "
        f"{r.crosscheck['rel_diff']:.2e}",
    ]
    if r.opt.EL_cv is not None:
        L.append(f"   Chẩn đoán dừng KKT xấp xỉ trên phần tử tự do = {r.opt.EL_cv:.3%}.")
        L.append("   Chỉ số này không kiểm tra đầy đủ nhân tử, bù trừ và điều kiện dừng của mọi ràng buộc bền/biên.")
    L += [
        "",
        "V. LƯU Ý VÀ GIỚI HẠN MÔ HÌNH",
        "   - Mô hình Euler–Bernoulli, bỏ qua biến dạng cắt và ổn định cục bộ/tải cục bộ.",
        "   - Tiết diện thực phải làm tròn theo thép hình/tấm có sẵn và chế tạo vút thon liên tục (hoặc theo đoạn).",
    ]
    return "\n".join(L)


def _struct_report(r: StructOptResult) -> str:
    inp = r.inp; o = r.opt; m = r.model
    L = [
        "=" * 56,
        f"   THUYẾT MINH — TỐI ƯU {'KHUNG PHẲNG' if inp.mode == 'frame' else 'DÀN THÉP'} (FEM + OC/FSD)",
        "=" * 56,
        "",
        "I. BÀI TOÁN",
        f"   Số nút = {len(inp.nodes)},  số thanh = {len(inp.members)},  "
        f"liên kết nút = {'hàn cứng (khung)' if inp.mode == 'frame' else 'khớp (dàn)'}",
        f"   Họ tiết diện: {inp.fam.name}  (n = {inp.fam.n:g})",
        f"   E = {inp.E:.4g} kN/m²,  [σ] = {inp.sigma_a / 1e3:.1f} MPa,  ρg = {inp.rho:.1f} kN/m³",
        f"   Ổn định Euler khi nén: {'bật' if inp.check_buckling else 'tắt'}  (k = {inp.k_eff:g}, FS = {inp.FS_buckling:g})",
        f"   Cận diện tích: {inp.A_min:.3g} ≤ A ≤ {inp.A_max:.3g} m²" +
        (f";  số đoạn chia mỗi thanh = {inp.segments}" if inp.mode == "frame" else ""),
        "   Ràng buộc chuyển vị: " + (", ".join(f"|{d}(N{n})| ≤ {lim * 1e3:.2f} mm" for n, d, lim in inp.disp_limits)
                                      if inp.disp_limits else "không"),
        "",
        "II. TRÌNH TỰ PHÂN TÍCH VÀ ĐẶT BÀI TOÁN",
        "1) Phân tích FEM: lắp ráp K(A), giải K(A)u = F; phục hồi N_e, M_e và chuyển vị u_k.",
        "2) Biến thiết kế là các diện tích A_e; thể tích vật liệu: V(A) = Σ_e A_e L_e.",
        "   Bài toán theo slide: min V(A). Solver tối thiểu W(A)=ρg·V(A); vì ρg là hằng số dương, nghiệm A* không đổi.",
        "3) Ràng buộc bền/ổn định từng phần tử:",
        "   g^σ_e(A) = |N_e|/(A_e σ_N,e) + |M_e|/(W_e[σ]) − 1 ≤ 0;",
        "   khi nén, σ_N,e = min([σ], σ_Euler,e). Ràng buộc chuyển vị: g^δ_k = |u_k|/ū_k − 1 ≤ 0.",
        "4) Ràng buộc biên: A_min ≤ A_e ≤ A_max; các ràng buộc hoạt động được kiểm tra tại nghiệm số.",
        "",
        "III. ĐIỀU KIỆN DỪNG/KKT VÀ ĐỘ NHẠY",
        "1) Lagrangian tổng quát theo mục tiêu thể tích: ℒ = V + Σ_e μ_e g^σ_e + Σ_k λ_k g^δ_k + các cận A, với μ_e, λ_k ≥ 0.",
        "2) Điều kiện KKT: dừng ∇_Aℒ=0; khả thi nguyên thủy g≤0; khả thi đối ngẫu μ,λ≥0;",
        "   bù trừ μ_e g^σ_e = 0 và λ_k g^δ_k = 0 (kèm nhân tử cận diện tích).",
        "3) Độ nhạy chuyển vị lấy bằng bài toán liên hợp K ū = e_k:",
        "   ∂u_k/∂A_e = −ūᵀ(∂K/∂A_e)u; đây là đại lượng dùng trong bước SLSQP/OC.",
        "4) Thuật toán thực thi: FEM → OC/FSD tạo thiết kế khả thi ban đầu → SLSQP tinh chỉnh ràng buộc chuyển vị.",
        "   Với dàn tĩnh định và chỉ ràng buộc thể tích, công thức Schmit/Maxwell–Mohr cho A_e ∝ |N_e|.",
        "",
        "IV. KẾT QUẢ VÀ KIỂM TRA",
        "",
        f"   Thể tích tối ưu V* = Σ A_eL_e = {_volume(o.A, m):.6f} m³.",
        f"   Thể tích tiết diện đều = A_đều·ΣL_e = {o.A_uniform * float(np.sum(m.L)):.6f} m³.",
        f"   Khối lượng tối ưu m* = W*/g₀ = {weight_to_mass_kg(r.weight):.2f} kg.",
        f"   Khối lượng tiết diện đều = {weight_to_mass_kg(r.weight_uniform):.2f} kg "
        f"(A đều = {o.A_uniform:.5g} m²).",
        f"   Trọng lượng tương ứng W* = ρgV* = {r.weight:.4f} kN  (ρg = {inp.rho:.4g} kN/m³).",
        f"   Tiết kiệm khối lượng   = {r.saving * 100:.2f}%",
        f"   Hội tụ: {'có' if o.converged else 'chưa'} sau {o.iters} vòng; thoả ràng buộc: {'có' if o.feasible else 'KHÔNG'}",
        f"   Utilization lớn nhất = {float(np.max(o.util)):.4f}",
    ]
    if o.disp_ratio.size:
        L.append("   Tỷ số chuyển vị |δ|/giới hạn = " + ", ".join(f"{v:.4f}" for v in o.disp_ratio))
    if o.EL_cv is not None:
        L.append(f"   Chẩn đoán cân bằng dừng trên phần tử tự do = {o.EL_cv:.3%}.")
        L.append("   Giới hạn chẩn đoán: không phải nghiệm nhân tử đầy đủ; chưa chứng nhận bù trừ cho từng ràng buộc bền và cận A.")
    L.append(f"   Khả thi nguyên thủy cuối: U_max = {float(np.max(o.util)):.5f} ≤ 1 và "
             f"δ_max/δ_giới_hạn = {float(np.max(o.disp_ratio)) if o.disp_ratio.size else 0.0:.5f} ≤ 1: "
             f"{'ĐẠT' if o.feasible else 'KHÔNG ĐẠT'}.")
    L.append("   Bộ giải hiện không xuất các nhân tử KKT cho từng ràng buộc hoạt động; không diễn giải phần dư dừng như chứng nhận KKT đầy đủ.")
    if r.crosscheck is not None:
        L.append(f"   Đối chiếu fem_core.solve_plane_frame: sai khác chuyển vị tương đối = "
                 f"{r.crosscheck['rel_diff']:.2e}")
    L += ["", "IV.1 BẢNG NỘI LỰC VÀ TIẾT DIỆN THEO THANH"]
    nm = len(inp.members)
    L.append(f"   {'Thanh':>5} {'i-j':>7} {'L(m)':>8} {'A(cm²)':>10} {'N(kN)':>10} {'Mmax(kNm)':>10} {'U':>7}")
    for k in range(nm):
        sel = r.member_of_el == k
        Lk = float(np.sum(m.L[sel]))
        Nk = float(o.N[sel][np.argmax(np.abs(o.N[sel]))])
        Mk = float(np.max(o.M[sel]))
        Uk = float(np.max(o.util[sel]))
        i, j = inp.members[k]
        L.append(f"   {k:>5} {str(i) + '-' + str(j):>7} {Lk:>8.3f} {r.A_member[k] * 1e4:>10.3f} "
                 f"{Nk:>10.3f} {Mk:>10.3f} {Uk:>7.3f}")
    L += ["", "IV.2 PHẢN LỰC GỐI"]
    names = ["Rx", "Ry", "Mz"] if inp.mode == "frame" else ["Rx", "Ry"]
    dpn = m.dof_per_node
    for d, v in sorted(r.reactions.items()):
        L.append(f"   Nút {d // dpn}: {names[d % dpn]} = {v:.4f}")
    L += ["", "V. GIẢ THIẾT VÀ GIỚI HẠN MÔ HÌNH",
          "   - FSD/OC cho nghiệm tốt thực hành; với kết cấu siêu tĩnh, nghiệm là cực trị địa phương.",
          "   - Chưa xét tải trọng bản thân, ổn định tổng thể bậc hai (P-Δ), mỏi và chi tiết liên kết.",
          "   - Tiết diện thực cần làm tròn theo thép hình/ống tiêu chuẩn rồi kiểm tra lại theo TCVN/AISC."]
    return "\n".join(L)
