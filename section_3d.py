"""3D lofts for before/after structural-member section sizing."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CROSS_SECTION_EXAGGERATION = 3.0


def _area_stations(area_profile, count: int) -> np.ndarray:
    areas = np.asarray(area_profile, dtype=float).reshape(-1)
    if areas.size == 0 or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("Profile diện tích phải gồm các giá trị dương, hữu hạn.")
    if areas.size == 1:
        return np.full(count, areas[0])
    return np.interp(np.linspace(0.0, 1.0, count), np.linspace(0.0, 1.0, areas.size), areas)


def _section_contours(family, area: float, circle_points: int = 24):
    dimensions = family.cross_section_dimensions(area)
    if family.shape_kind == "rect":
        width, height = dimensions["b"], dimensions["h"]
        return [np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])], None

    if family.shape_kind in ("circle", "tube"):
        radius = dimensions["D"] / 2
        angle = np.linspace(0.0, 2 * math.pi, circle_points, endpoint=False)
        outer = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        if family.shape_kind == "circle":
            return [outer], None
        inner_radius = radius - dimensions["t"]
        inner = np.column_stack([inner_radius * np.cos(angle), inner_radius * np.sin(angle)])
        return [outer], inner

    if family.shape_kind == "ibeam":
        height, width, thickness = dimensions["h"], dimensions["b_f"], dimensions["t_f"]
        contours = []
        for center_z in (-height / 2 + thickness / 2, height / 2 - thickness / 2):
            contours.append(np.array([
                [-width / 2, center_z - thickness / 2],
                [width / 2, center_z - thickness / 2],
                [width / 2, center_z + thickness / 2],
                [-width / 2, center_z + thickness / 2],
            ]))
        return contours, None

    side = math.sqrt(area)
    return [np.array([[-side / 2, -side / 2], [side / 2, -side / 2],
                      [side / 2, side / 2], [-side / 2, side / 2]])], None


def _solid_loft(length: float, rings: list[np.ndarray], color: str, name: str) -> go.Mesh3d:
    station_count = len(rings)
    vertex_count = len(rings[0])
    x_values = np.repeat(np.linspace(0.0, length, station_count), vertex_count).tolist()
    y_values = np.concatenate(rings)[:, 0].tolist()
    z_values = np.concatenate(rings)[:, 1].tolist()
    face_i: list[int] = []
    face_j: list[int] = []
    face_k: list[int] = []

    for station in range(station_count - 1):
        for point in range(vertex_count):
            next_point = (point + 1) % vertex_count
            lower_a = station * vertex_count + point
            lower_b = station * vertex_count + next_point
            upper_a = (station + 1) * vertex_count + point
            upper_b = (station + 1) * vertex_count + next_point
            face_i.extend([lower_a, lower_b])
            face_j.extend([lower_b, upper_b])
            face_k.extend([upper_a, upper_a])

    start_center = len(x_values)
    x_values.append(0.0)
    y_values.append(float(np.mean(rings[0][:, 0])))
    z_values.append(float(np.mean(rings[0][:, 1])))
    end_center = len(x_values)
    x_values.append(length)
    y_values.append(float(np.mean(rings[-1][:, 0])))
    z_values.append(float(np.mean(rings[-1][:, 1])))
    for point in range(vertex_count):
        next_point = (point + 1) % vertex_count
        face_i.extend([start_center, end_center])
        face_j.extend([next_point, (station_count - 1) * vertex_count + point])
        face_k.extend([point, (station_count - 1) * vertex_count + next_point])

    return go.Mesh3d(x=x_values, y=y_values, z=z_values, i=face_i, j=face_j, k=face_k,
                     color=color, opacity=0.94, flatshading=False, name=name,
                     lighting=dict(ambient=0.85, diffuse=0.45, specular=0.06, roughness=0.9, fresnel=0.0),
                     lightposition=dict(x=100, y=200, z=100),
                     hovertemplate=f"{name}<extra></extra>")


def _tube_loft(length: float, outer_rings: list[np.ndarray], inner_rings: list[np.ndarray],
               color: str, name: str) -> go.Mesh3d:
    station_count = len(outer_rings)
    vertex_count = len(outer_rings[0])
    x_values = np.repeat(np.linspace(0.0, length, station_count), vertex_count * 2).tolist()
    y_values = np.concatenate([np.concatenate([outer[:, 0], inner[:, 0]])
                               for outer, inner in zip(outer_rings, inner_rings)]).tolist()
    z_values = np.concatenate([np.concatenate([outer[:, 1], inner[:, 1]])
                               for outer, inner in zip(outer_rings, inner_rings)]).tolist()
    face_i: list[int] = []
    face_j: list[int] = []
    face_k: list[int] = []

    for station in range(station_count - 1):
        base = station * vertex_count * 2
        next_base = (station + 1) * vertex_count * 2
        for point in range(vertex_count):
            next_point = (point + 1) % vertex_count
            oa, ob = base + point, base + next_point
            na, nb = next_base + point, next_base + next_point
            ia, ib = base + vertex_count + point, base + vertex_count + next_point
            nia, nib = next_base + vertex_count + point, next_base + vertex_count + next_point
            face_i.extend([oa, ob, ia, ib])
            face_j.extend([ob, nb, nia, nia])
            face_k.extend([na, na, ib, nib])

    for point in range(vertex_count):
        next_point = (point + 1) % vertex_count
        inner = vertex_count + point
        inner_next = vertex_count + next_point
        end = (station_count - 1) * vertex_count * 2
        end_inner = end + vertex_count + point
        end_inner_next = end + vertex_count + next_point
        face_i.extend([point, point, end + point, end + point])
        face_j.extend([inner_next, inner, end + next_point, end_inner_next])
        face_k.extend([next_point, inner_next, end_inner_next, end_inner])

    return go.Mesh3d(x=x_values, y=y_values, z=z_values, i=face_i, j=face_j, k=face_k,
                     color=color, opacity=0.94, flatshading=False, name=name,
                     lighting=dict(ambient=0.85, diffuse=0.45, specular=0.06, roughness=0.9, fresnel=0.0),
                     lightposition=dict(x=100, y=200, z=100),
                     hovertemplate=f"{name}<extra></extra>")


def _section_outline_traces(family, length: float, area_profile, color: str,
                            station_count: int = 25) -> list[go.Scatter3d]:
    areas = _area_stations(area_profile, station_count)
    station_indices = np.linspace(0, station_count - 1, 5, dtype=int)
    traces = []
    for station in station_indices:
        contours, inner = _section_contours(family, float(areas[station]))
        if inner is not None:
            contours.append(inner)
        x_value = length * station / (station_count - 1)
        for contour in contours:
            scaled = contour * CROSS_SECTION_EXAGGERATION
            closed = np.vstack([scaled, scaled[0]])
            traces.append(go.Scatter3d(
                x=np.full(len(closed), x_value), y=closed[:, 0], z=closed[:, 1],
                mode="lines", line=dict(color=color, width=2), opacity=0.8,
                hoverinfo="skip", showlegend=False,
            ))
    return traces


def _member_meshes(family, length: float, area_profile, color: str, name: str,
                   station_count: int = 25) -> list[go.Mesh3d]:
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Chiều dài cấu kiện phải dương và hữu hạn.")
    areas = _area_stations(area_profile, station_count)
    contours_by_station = [_section_contours(family, float(area)) for area in areas]
    contours_by_station = [([contour * CROSS_SECTION_EXAGGERATION for contour in contours[0]],
                             contours[1] * CROSS_SECTION_EXAGGERATION if contours[1] is not None else None)
                            for contours in contours_by_station]
    meshes = []

    if contours_by_station[0][1] is not None:
        outer_rings = [contours[0][0] for contours in contours_by_station]
        inner_rings = [contours[1] for contours in contours_by_station]
        mesh = _tube_loft(length, outer_rings, inner_rings, color, name)
        return [mesh, *_section_outline_traces(family, length, area_profile,
                                                "#263b50" if color == "#94a3b8" else "#063b63",
                                                station_count)]

    contour_count = len(contours_by_station[0][0])
    meshes = []
    for contour_index in range(contour_count):
        rings = [contours[0][contour_index] for contours in contours_by_station]
        mesh_name = f"{name} · cánh {contour_index + 1}" if family.shape_kind == "ibeam" else name
        meshes.append(_solid_loft(length, rings, color, mesh_name))
    meshes.extend(_section_outline_traces(family, length, area_profile,
                                         "#263b50" if color == "#94a3b8" else "#063b63",
                                         station_count))
    return meshes


def section_3d_comparison(family, length: float, area_before, area_after,
                          orientation: str = "horizontal") -> go.Figure:
    """Build same-scale 3D solids for the uniform reference and optimized member."""
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("Hướng cấu kiện phải là 'horizontal' hoặc 'vertical'.")
    before_areas = _area_stations(area_before, 25)
    after_areas = _area_stations(area_after, 25)
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                           subplot_titles=(f"Trước · A = {before_areas[0] * 1e4:.2f} cm²",
                                          f"Sau · A = {after_areas.min() * 1e4:.2f}–{after_areas.max() * 1e4:.2f} cm²"))

    for trace in _member_meshes(family, length, area_before, "#94a3b8", "Trước tối ưu"):
        figure.add_trace(trace, row=1, col=1)
    for trace in _member_meshes(family, length, area_after, "#1976b9", "Sau tối ưu"):
        figure.add_trace(trace, row=1, col=2)

    if orientation == "vertical":
        for trace in figure.data:
            old_x, old_y, old_z = trace.x, trace.y, trace.z
            trace.x, trace.y, trace.z = old_y, old_z, old_x
        scene = dict(xaxis_title="Rộng (×3)", yaxis_title="Cao (×3)", zaxis_title="Chiều cao cột thật (m)",
                     aspectmode="data", bgcolor="rgba(0,0,0,0)",
                     camera=dict(eye=dict(x=1.55, y=1.1, z=2.0)))
    else:
        scene = dict(xaxis_title="Chiều dài thật (m)", yaxis_title="Rộng (×3)", zaxis_title="Cao (×3)",
                     aspectmode="data", bgcolor="rgba(0,0,0,0)",
                     camera=dict(eye=dict(x=0.8, y=1.7, z=1.2)))
    figure.update_layout(height=500, margin=dict(l=0, r=0, t=60, b=0),
                         paper_bgcolor="rgba(0,0,0,0)", scene=scene, scene2=scene)
    return figure


def section_3d_note(family) -> str:
    if family.shape_kind == "ibeam":
        note = "Mô hình chữ I chỉ tạo hai cánh; bụng được bỏ qua như trong quan hệ I(A)."
    elif family.shape_kind == "custom":
        note = "Họ tùy chỉnh chưa có hình học vật lý; 3D dùng tiết diện vuông tương đương A để minh họa, không phải cấu tạo thật."
    elif family.shape_kind == "tube":
        note = "Ống được dựng theo gần đúng thành mỏng A ≈ πDt."
    else:
        note = "Hình khối dựng theo hình học và diện tích của họ tiết diện đã chọn."
    return (f"{note} Rộng/cao phóng đại ×{CROSS_SECTION_EXAGGERATION:g} để dễ nhìn; "
            "chiều dài giữ đúng tỷ lệ. Các vòng viền đánh dấu năm mặt cắt dọc cấu kiện.")