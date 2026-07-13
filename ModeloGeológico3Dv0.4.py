import os
import re
import time
import json
import webbrowser
import traceback
import numpy as np
import pandas as pd
import pyvista as pv
import geopandas as gpd
import matplotlib.path as mpath

from shapely.geometry import Point
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

# ============================================================
# CONFIGURACION
# ============================================================

CALICATAS_FILE = "calicatas.csv"
TOPO_FILE = "topografia.txt"
ROI_FILE = "roi.geojson"
DESLINDE_FILE = "deslinde.geojson"
EDIFICACION_FILE = "edificacion.geojson"
EXPORT_HTML = "visor_geologico_dual.html"

GRID_SIZE = 150
HORIZONTAL_ANISOTROPY = 15.0
STRUCTURAL_DIP_ANGLE = 0.0 

# Interpolación exacta
RBF_SMOOTHING = 0.0
MIN_POINTS_PER_UNIT = 1

# ============================================================
# LITOLOGIAS Y DOMINIOS ACTUALIZADOS
# ============================================================
GEO_DICT = {
    # Modelo Geológico
    "SN":  {"color": "#D2691E", "name": "Horizonte superficial"},
    "RA":  {"color": "#A9A9A9", "name": "Horizonte de relleno"},
    "HA":  {"color": "#FFD700", "name": "Horizonte sedimentario natural"},
    "HM":  {"color": "#FFA500", "name": "Horizonte cementado"},
    "FC":  {"color": "#F0E68C", "name": "Facies carbonatadas y bioclásticas"},
    "HG":  {"color": "#CD853F", "name": "Horizonte gravoso"},
    
    # Modelo Geotécnico (Nuevo planteamiento)
    "CI":  {"color": "#9E9E9E", "name": "Cobertura Incompetente"},
    "NAC": {"color": "#FFEB3B", "name": "Nivel Arenoso Competente"},
    "NCC": {"color": "#FF9800", "name": "Nivel Cementado Carbonatado"}
}

COVER_LITHOS_GEO = ["RA", "SN"]
COVER_LITHOS_GEOTEC = ["CI"]

def get_geo_prop(code, prop):
    fallback = {"color": "#FFFFFF", "name": f"Desconocido ({code})"}
    return GEO_DICT.get(code, fallback)[prop]

def get_layer_order(unit_str):
    geo = unit_str.split('_')[1] if '_' in unit_str else unit_str
    order = {"RA": 10, "SN": 20, "HA": 30, "HM": 40, "FC": 50, "HG": 60,
             "CI": 10, "NAC": 30, "NCC": 40}
    return order.get(geo, 99)

# ============================================================
# LOGGER Y UTILIDADES
# ============================================================
START_TIME = time.time()

def log(msg):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:8.1f}s] {msg}", flush=True)

def log_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70, flush=True)

def parse_number(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (float, int)): return float(value)
    value = str(value).strip().replace(",", ".")
    try: return float(value)
    except: return np.nan

def polydata_to_dict(poly):
    if not poly.is_all_triangles: poly = poly.triangulate()
    vertices = np.round(poly.points, 3).flatten().tolist()
    faces = poly.faces
    indices = []
    i = 0
    while i < len(faces):
        n = faces[i]
        if n == 3: indices.extend([int(faces[i+1]), int(faces[i+2]), int(faces[i+3])])
        i += n + 1
    return {"vertices": vertices, "indices": indices}

def lines_to_dict(poly):
    vertices = np.round(poly.points, 3).flatten().tolist()
    lines = poly.lines
    indices = []
    i = 0
    while i < len(lines):
        n = lines[i]
        for j in range(n - 1): indices.extend([int(lines[i + 1 + j]), int(lines[i + 2 + j])])
        i += n + 1
    return {"vertices": vertices, "indices": indices}

def load_roi(roi_file):
    log(f"Cargando ROI: {roi_file}")
    gdf = gpd.read_file(roi_file)
    return gdf.geometry.iloc[0]

def create_roi_mask(xx, yy, roi_geom):
    if roi_geom.geom_type == 'Polygon': polygons = [roi_geom]
    elif roi_geom.geom_type == 'MultiPolygon': polygons = list(roi_geom.geoms)
    else: return np.ones(xx.shape, dtype=bool)
        
    points = np.column_stack((xx.ravel(), yy.ravel()))
    mask = np.zeros(len(points), dtype=bool)
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        path = mpath.Path(np.column_stack((x, y)))
        poly_mask = path.contains_points(points)
        for interior in poly.interiors:
            ix, iy = interior.coords.xy
            ipath = mpath.Path(np.column_stack((ix, iy)))
            poly_mask &= ~ipath.contains_points(points)
        mask |= poly_mask
    return mask.reshape(xx.shape)

def filter_points_by_roi(df, roi_geom):
    mask = [roi_geom.contains(Point(row["X"], row["Y"])) for _, row in df.iterrows()]
    return df[mask]

# ============================================================
# PROYECCIÓN Y MATEMÁTICA RBF ESTRICTA
# ============================================================
def process_projected_vector(filepath, topo_rbf_model, is_building=False, is_deslinde=False, z_offset=1.0):
    if not os.path.exists(filepath): return []
    log(f"Proyectando vectores e infraestructuras: {filepath}")
    gdf = gpd.read_file(filepath)
    features = []
    for geom in gdf.geometry:
        if geom is None: continue
        parts = [geom] if geom.geom_type in ['Polygon', 'LineString'] else list(geom.geoms)
        for part in parts:
            if part.geom_type == 'Polygon':
                x, y = part.exterior.coords.xy
                pts = np.column_stack([np.array(x)/HORIZONTAL_ANISOTROPY, np.array(y)/HORIZONTAL_ANISOTROPY])
                z_vals = topo_rbf_model(pts)
                z_avg = float(np.mean(z_vals)) + z_offset
                features.append({
                    "type": "deslinde" if is_deslinde else ("polygon" if is_building else "line"),
                    "outline": [[float(ix), float(iy)] for ix, iy in zip(x, y)],
                    "coords": [[float(ix), float(iy), float(iz) + z_offset] for ix, iy, iz in zip(x, y, z_vals)],
                    "zBase": z_avg,
                    "height": 5.0 if is_building else 0.0
                })
            elif part.geom_type == 'LineString':
                x, y = part.coords.xy
                pts = np.column_stack([np.array(x)/HORIZONTAL_ANISOTROPY, np.array(y)/HORIZONTAL_ANISOTROPY])
                z_vals = topo_rbf_model(pts)
                features.append({
                    "type": "line",
                    "coords": [[float(ix), float(iy), float(iz) + z_offset] for ix, iy, iz in zip(x, y, z_vals)]
                })
    return features

def load_calicatas(csv_file):
    df = pd.read_csv(csv_file, sep=";")
    df["E"], df["N"], df["ELEV"] = df["E"].apply(parse_number), df["N"].apply(parse_number), df["ELEV"].apply(parse_number)
    estratos = []
    horizon_regex = re.compile(r"H(\d+)_GEO")
    h_nums = sorted(list(set([int(m.group(1)) for c in df.columns if (m := horizon_regex.match(c))])))

    for _, row in df.iterrows():
        x, y, surface_z = row["E"], row["N"], row["ELEV"]
        if np.isnan(x) or np.isnan(y): continue
        for h in h_nums:
            geo = row.get(f"H{h}_GEO")
            geotec = row.get(f"H{h}_GEOTEC")
            if pd.isna(geotec) and h == 4 and "HR_GEOTEC" in row:
                geotec = row.get("HR_GEOTEC")

            if pd.isna(geo) and pd.isna(geotec): continue
            
            techo, base = parse_number(row.get(f"H{h}_TECHO")), parse_number(row.get(f"H{h}_BASE"))
            if np.isnan(techo) or np.isnan(base): continue
            
            estratos.append({
                "ID": row["IDENTIFICA"], "X": x, "Y": y, "SURFACE": surface_z,
                "TOP_Z": surface_z - techo, "BASE_Z": surface_z - base, 
                "THICKNESS": base - techo, "H_NUM": f"H{h}",
                "GEO": str(geo).strip() if pd.notna(geo) else "UNKNOWN",
                "GEOTEC": str(geotec).strip() if pd.notna(geotec) else "UNKNOWN"
            })
    return pd.DataFrame(estratos)

def load_topography(file_path):
    topo = pd.read_csv(file_path, delim_whitespace=True, header=None).iloc[:, :3]
    topo.columns = ["X", "Y", "Z"]
    topo["X"], topo["Y"], topo["Z"] = topo["X"].apply(parse_number), topo["Y"].apply(parse_number), topo["Z"].apply(parse_number)
    return topo.dropna()

def get_rbf_model(sample_x, sample_y, sample_z):
    df = pd.DataFrame({'x': sample_x, 'y': sample_y, 'z': sample_z})
    df = df.groupby(['x', 'y']).mean().reset_index()
    
    scaled_x, scaled_y = df['x'].values / HORIZONTAL_ANISOTROPY, df['y'].values / HORIZONTAL_ANISOTROPY
    pts_known = np.column_stack([scaled_x, scaled_y])
    sz = df['z'].values
    n_pts = len(sz)
    
    if n_pts == 1:
        return lambda p: np.full(p.shape[0], sz[0])
    else:
        try:
            if n_pts < 150:
                rbf = RBFInterpolator(pts_known, sz, kernel="linear", smoothing=RBF_SMOOTHING)
            else:
                k_neighbors = min(15, n_pts)
                rbf = RBFInterpolator(pts_known, sz, kernel="linear", smoothing=RBF_SMOOTHING, neighbors=k_neighbors)
        except Exception:
            if n_pts < 150:
                rbf = RBFInterpolator(pts_known, sz, kernel="linear", smoothing=0.001)
            else:
                k_neighbors = min(15, n_pts)
                rbf = RBFInterpolator(pts_known, sz, kernel="linear", smoothing=0.001, neighbors=k_neighbors)
        return lambda p: rbf(p)

def calculate_structural_trend(topo_df):
    log("Calculando tendencia geomorfológica y estructural...")
    A = np.c_[topo_df["X"], topo_df["Y"], np.ones(len(topo_df))]
    C, _, _, _ = np.linalg.lstsq(A, topo_df["Z"], rcond=None)
    grad_x, grad_y = C[0], C[1]
    grad_mag = np.hypot(grad_x, grad_y)
    u_x, u_y = grad_x / grad_mag, grad_y / grad_mag
    dip_slope = np.tan(np.radians(STRUCTURAL_DIP_ANGLE)) 
    return u_x, u_y, dip_slope

# ============================================================
# MOTOR DE CONSTRUCCIÓN ESTRATIGRÁFICA (TU MATEMÁTICA INTACTA)
# ============================================================
def build_stratigraphic_model(estratos_df_full, mode, xx, yy, pts, pts_unscaled, roi_mask, topo_z, topo_gradient_mag, u_x, u_y, dip_slope, all_bhs_df, bh_bottoms):
    
    log(f"--- INICIANDO INTERPOLACIÓN MODO: {mode} ---")
    estratos_df = estratos_df_full[estratos_df_full[mode] != "UNKNOWN"].copy()
    
    # === NUEVO BLOQUE: FUSIÓN INTELIGENTE DE ESTRATOS CONSECUTIVOS ===
    # 1. Ordenamos por calicata y profundidad (de arriba hacia abajo)
    estratos_df = estratos_df.sort_values(["ID", "TOP_Z"], ascending=[True, False]).reset_index(drop=True)
    
    # 2. Identificamos si la capa actual es igual a la capa anterior (Ej: CI seguido de CI)
    estratos_df['block'] = (estratos_df[mode] != estratos_df[mode].shift()).cumsum()
    
    # 3. Fusionamos los estratos consecutivos sumando sus espesores
    estratos_df = estratos_df.groupby(['ID', 'block']).agg({
        'X': 'first', 'Y': 'first', 'SURFACE': 'first',
        'TOP_Z': 'max',            # Nos quedamos con el techo más alto
        'BASE_Z': 'min',           # Nos quedamos con la base más profunda
        'THICKNESS': 'sum',        # Sumamos el espesor total
        'H_NUM': 'first',          # Mantenemos el índice superior
        mode: 'first'
    }).reset_index()
    # =================================================================
    
    estratos_df["UNIT_KEY"] = estratos_df["H_NUM"] + "_" + estratos_df[mode]
    estratos_df["LITHO"] = estratos_df[mode]
    
    units_keys = estratos_df["UNIT_KEY"].unique()
    cover_lithos = COVER_LITHOS_GEO if mode == "GEO" else COVER_LITHOS_GEOTEC
    
    cover_units = [u for u in units_keys if any(c in u for c in cover_lithos)]
    base_units = [u for u in units_keys if not any(c in u for c in cover_lithos)]
    
    cover_units.sort(key=get_layer_order)
    base_units.sort(key=lambda u: estratos_df[estratos_df["UNIT_KEY"] == u]["TOP_Z"].mean(), reverse=True)

    current_roof_grid = topo_z.copy()
    estratos_export = []
    global_order_counter = 0
    dx, dy = abs(xx[0, 1] - xx[0, 0]), abs(yy[1, 0] - yy[0, 0])

    # FASE 1: COBERTURA
    ra_presence_mask = np.zeros(xx.shape, dtype=bool)

    for unit in cover_units:
        sub = estratos_df[estratos_df["UNIT_KEY"] == unit]
        if len(sub) < MIN_POINTS_PER_UNIT: continue
        
        litho = sub.iloc[0]["LITHO"]
        log(f" > Cobertura: {unit} ({len(sub)} pts)")
        
        x, y, thickness = sub["X"].values, sub["Y"].values, sub["THICKNESS"].values
        top_grid = current_roof_grid.copy()

        pos_ids = sub["ID"].unique()
        neg_df = all_bhs_df[~all_bhs_df["ID"].isin(pos_ids)]
        if not neg_df.empty:
            kdtree_pos = cKDTree(np.column_stack([x, y]))
            kdtree_neg = cKDTree(neg_df[["X", "Y"]].values)
            d_pos, _ = kdtree_pos.query(pts_unscaled)
            d_neg, _ = kdtree_neg.query(pts_unscaled)
            w = d_neg / (d_pos + d_neg + 1e-9)
            pinch_out = (3 * (w**2) - 2 * (w**3)).reshape(xx.shape)
        else:
            pinch_out = np.ones(xx.shape)
        
        rbf_thick = get_rbf_model(x, y, thickness)
        thick_raw = np.maximum(rbf_thick(pts).reshape(xx.shape), 0)
        
        if litho == "RA":
            dist_tree = cKDTree(np.column_stack([x, y]))
            dist_grid, _ = dist_tree.query(pts_unscaled, k=1)
            dist_grid = dist_grid.reshape(xx.shape)
            ra_fade = np.clip(1.0 - (dist_grid - 10.0) / 20.0, 0.0, 1.0)
            thick_raw *= ra_fade
            thick_raw *= np.clip(1.0 - (topo_gradient_mag * 5.0), 0.2, 1.0)
            
            thick_final = thick_raw * pinch_out
            thick_final = np.where(thick_final > 0.01, thick_final, 0.0)
            ra_presence_mask |= (thick_final > 0.01)
            
        elif litho == "CI":
            dist_tree = cKDTree(np.column_stack([x, y]))
            dist_grid, _ = dist_tree.query(pts_unscaled, k=1)
            dist_grid = dist_grid.reshape(xx.shape)
            ra_fade = np.clip(1.0 - (dist_grid - 10.0) / 20.0, 0.0, 1.0)
            
            base_sn = 0.5
            excess = np.maximum(thick_raw - base_sn, 0.0)
            excess *= ra_fade
            excess *= np.clip(1.0 - (topo_gradient_mag * 5.0), 0.2, 1.0)
            
            thick_raw = base_sn + excess
            thick_final = thick_raw * pinch_out
            thick_final = np.where(thick_final > 0.01, thick_final, 0.0)

        elif litho == "SN":
            thick_raw = np.minimum(thick_raw, 0.5)
            thick_final = thick_raw * pinch_out
            thick_final[ra_presence_mask] = 0.0
            thick_final = np.where(thick_final > 0.01, thick_final, 0.0)
        else:
            thick_final = thick_raw * pinch_out

        base_grid = top_grid - thick_final
        valid_mask = thick_final >= 0.01
        mask_final = roi_mask & valid_mask

        if mask_final.sum() >= 5:
            volumen_m3 = np.sum(thick_final[mask_final]) * dx * dy
            grid = pv.StructuredGrid(np.dstack((xx, xx)), np.dstack((yy, yy)), np.dstack((base_grid, top_grid)))
            grid.point_data["valid"] = np.dstack((mask_final, mask_final)).flatten(order='F').astype(int)
            surface = grid.threshold(0.5, scalars="valid").extract_surface().triangulate()
            
            if surface.n_points > 0:
                mesh_data = polydata_to_dict(surface)
                mesh_data["name"] = unit
                mesh_data["geo"] = litho
                mesh_data["color"] = get_geo_prop(litho, "color")
                mesh_data["volumen"] = float(volumen_m3)
                mesh_data["order"] = global_order_counter
                global_order_counter += 10
                estratos_export.append(mesh_data)

        current_roof_grid = np.where(valid_mask, base_grid, top_grid)

    unconformity_grid = current_roof_grid.copy()
    
    # FASE 2: BASAMENTO (ESTRICTAMENTE TU LOGICA)
    log("Fase 2: Estratos Subparalelos (Basamento)")
    current_basement_roof = unconformity_grid.copy()
    trend_at_grid = -dip_slope * (xx * u_x + yy * u_y)

    for unit in base_units:
        sub = estratos_df[estratos_df["UNIT_KEY"] == unit]
        if len(sub) < MIN_POINTS_PER_UNIT: continue
        
        litho = sub.iloc[0]["LITHO"]
        log(f" > Basamento/Estratificado: {unit} ({len(sub)} pts)")
        
        x, y = sub["X"].values, sub["Y"].values
        thickness, top_z_vals = sub["THICKNESS"].values, sub["TOP_Z"].values

        trend_at_calicatas = -dip_slope * (x * u_x + y * u_y)
        residuals_top = top_z_vals - trend_at_calicatas
        rbf_top = get_rbf_model(x, y, residuals_top)
        
        pos_ids = sub["ID"].unique()
        true_neg_ids = []
        
        for bid in all_bhs_df["ID"]:
            if bid in pos_ids: continue
            bx = all_bhs_df.loc[all_bhs_df["ID"] == bid, "X"].values[0]
            by = all_bhs_df.loc[all_bhs_df["ID"] == bid, "Y"].values[0]
            bbot = bh_bottoms.get(bid, np.inf)
            
            btrend = -dip_slope * (bx * u_x + by * u_y)
            bpt = np.array([[bx / HORIZONTAL_ANISOTROPY, by / HORIZONTAL_ANISOTROPY]])
            btop_expected = float(rbf_top(bpt)[0]) + btrend
            
            if bbot < (btop_expected - 0.1):
                true_neg_ids.append(bid)
        
        neg_df = all_bhs_df[all_bhs_df["ID"].isin(true_neg_ids)]

        if not neg_df.empty:
            kdtree_pos = cKDTree(np.column_stack([x, y]))
            kdtree_neg = cKDTree(neg_df[["X", "Y"]].values)
            d_pos, _ = kdtree_pos.query(pts_unscaled)
            d_neg, _ = kdtree_neg.query(pts_unscaled)
            w = d_neg / (d_pos + d_neg + 1e-9)
            pinch_out = (3 * (w**2) - 2 * (w**3)).reshape(xx.shape)
        else:
            pinch_out = np.ones(xx.shape)

        top_raw = rbf_top(pts).reshape(xx.shape) + trend_at_grid
        
        if not neg_df.empty:
            x_thick = np.concatenate([x, neg_df["X"].values])
            y_thick = np.concatenate([y, neg_df["Y"].values])
            t_thick = np.concatenate([thickness, np.zeros(len(neg_df))])
        else:
            x_thick, y_thick, t_thick = x, y, thickness

        rbf_thick = get_rbf_model(x_thick, y_thick, t_thick)
        thick_raw = np.maximum(rbf_thick(pts).reshape(xx.shape), 0)

        base_raw = top_raw - (thick_raw * pinch_out)
        
        top_final = np.minimum(top_raw, current_basement_roof)
        base_final = np.minimum(base_raw, top_final - 0.001)
        thick_final = top_final - base_final
        
        valid_mask = thick_final >= 0.01
        mask_final = roi_mask & valid_mask

        if mask_final.sum() >= 5:
            volumen_m3 = np.sum(thick_final[mask_final]) * dx * dy
            grid = pv.StructuredGrid(np.dstack((xx, xx)), np.dstack((yy, yy)), np.dstack((base_final, top_final)))
            grid.point_data["valid"] = np.dstack((mask_final, mask_final)).flatten(order='F').astype(int)
            surface = grid.threshold(0.5, scalars="valid").extract_surface().triangulate()
            
            if surface.n_points > 0:
                mesh_data = polydata_to_dict(surface)
                mesh_data["name"] = unit
                mesh_data["geo"] = litho
                mesh_data["color"] = get_geo_prop(litho, "color")
                mesh_data["volumen"] = float(volumen_m3)
                mesh_data["order"] = global_order_counter
                global_order_counter += 10
                estratos_export.append(mesh_data)

        current_basement_roof = np.where(valid_mask, base_final, current_basement_roof)

    calicatas_export = []
    for borehole_id, sub in estratos_df.groupby("ID"):
        sub = sub.sort_values("TOP_Z", ascending=False)
        cx, cy = sub.iloc[0]["X"], sub.iloc[0]["Y"]
        top_surf = sub["SURFACE"].iloc[0]
        prof_total = top_surf - sub["BASE_Z"].min()
        segments = []
        for _, row in sub.iterrows():
            segments.append({
                "geoCode": row["LITHO"], 
                "geoName": get_geo_prop(row["LITHO"], "name"),
                "color": get_geo_prop(row["LITHO"], "color"),
                "zTop": row["TOP_Z"], "zBase": row["BASE_Z"]
            })
        calicatas_export.append({
            "id": str(borehole_id), "x": cx, "y": cy, 
            "zTop": top_surf, "profTotal": prof_total, "segments": segments
        })

    return estratos_export, calicatas_export

# ============================================================
# MAIN
# ============================================================
def main():
    try:
        log_section("COMPILANDO MODELO DUAL (GEOLÓGICO Y GEOTÉCNICO)")
        roi_geom = load_roi(ROI_FILE)
        bounds = roi_geom.bounds
        center_x, center_y = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        
        dx = (bounds[2] - bounds[0]) * 0.02
        dy = (bounds[3] - bounds[1]) * 0.02
        xi = np.linspace(bounds[0] - dx, bounds[2] + dx, GRID_SIZE)
        yi = np.linspace(bounds[1] - dy, bounds[3] + dy, GRID_SIZE)
        xx, yy = np.meshgrid(xi, yi)

        estratos_df = load_calicatas(CALICATAS_FILE)
        estratos_df = filter_points_by_roi(estratos_df, roi_geom)

        all_bhs_df = estratos_df.groupby("ID").first()[["X", "Y"]].reset_index()
        calicatas_coords = all_bhs_df[["X", "Y"]].values
        bh_bottoms = estratos_df.groupby("ID")["BASE_Z"].min()

        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        used_indices = set()
        for cx, cy in calicatas_coords:
            dist_sq = (xx_flat - cx)**2 + (yy_flat - cy)**2
            for idx in used_indices: dist_sq[idx] = np.inf
            min_idx = np.argmin(dist_sq)
            xx_flat[min_idx] = cx
            yy_flat[min_idx] = cy
            used_indices.add(min_idx)
            
        xx = xx_flat.reshape(xx.shape)
        yy = yy_flat.reshape(yy.shape)
        
        pts_unscaled = np.column_stack([xx.ravel(), yy.ravel()])
        pts = np.column_stack([xx.ravel() / HORIZONTAL_ANISOTROPY, yy.ravel() / HORIZONTAL_ANISOTROPY])

        topo_df = load_topography(TOPO_FILE)
        topo_df = filter_points_by_roi(topo_df, roi_geom)
        
        calicatas_collar = estratos_df.groupby("ID").first()[["X", "Y", "SURFACE"]].reset_index()
        calicatas_collar.rename(columns={"SURFACE": "Z"}, inplace=True)
        topo_df = pd.concat([topo_df, calicatas_collar[["X", "Y", "Z"]]], ignore_index=True)
        topo_df = topo_df.groupby(['X', 'Y']).mean().reset_index()
        
        log_section("PROCESANDO TOPOGRAFIA Y CURVAS DE NIVEL")
        topo_rbf_model = get_rbf_model(topo_df["X"].values, topo_df["Y"].values, topo_df["Z"].values)
        topo_z = topo_rbf_model(pts).reshape(xx.shape)
        
        gy, gx = np.gradient(topo_z, dy, dx)
        topo_gradient_mag = np.hypot(gx, gy)
        roi_mask = create_roi_mask(xx, yy, roi_geom)

        valid_pts = np.column_stack([xx[roi_mask], yy[roi_mask], topo_z[roi_mask]])
        topo_surface = pv.PolyData(valid_pts).delaunay_2d()
        topo_data = polydata_to_dict(topo_surface)
        
        z_min_topo, z_max_topo = np.min(topo_z[roi_mask]), np.max(topo_z[roi_mask])
        topo_surface["elevation"] = topo_surface.points[:, 2]
        contours = topo_surface.contour(isosurfaces=15, scalars="elevation")
        contours_data = lines_to_dict(contours)

        log_section("CONSTRUYENDO COLUMNAS ESTRATIGRÁFICAS")
        u_x, u_y, dip_slope = calculate_structural_trend(topo_df)
        
        estratos_geo_exp, calicatas_geo_exp = build_stratigraphic_model(estratos_df, "GEO", xx, yy, pts, pts_unscaled, roi_mask, topo_z, topo_gradient_mag, u_x, u_y, dip_slope, all_bhs_df, bh_bottoms)
        estratos_geotec_exp, calicatas_geotec_exp = build_stratigraphic_model(estratos_df, "GEOTEC", xx, yy, pts, pts_unscaled, roi_mask, topo_z, topo_gradient_mag, u_x, u_y, dip_slope, all_bhs_df, bh_bottoms)

        log_section("PROCESANDO INFRAESTRUCTURA (VECTORES)")
        deslinde_data = process_projected_vector(DESLINDE_FILE, topo_rbf_model, is_building=False, is_deslinde=True, z_offset=0.2)
        edific_data = process_projected_vector(EDIFICACION_FILE, topo_rbf_model, is_building=True, z_offset=0.1)

        log_section("EMPAQUETANDO APLICACION WEB (HTML)")
        
        app_data = {
            "center": {"x": center_x, "y": center_y, "z": np.mean(topo_z[roi_mask])},
            "bounds": {"xMin": bounds[0], "xMax": bounds[2], "yMin": bounds[1], "yMax": bounds[3], "zMin": z_min_topo - 50, "zMax": z_max_topo + 20},
            "topography": topo_data,
            "contours": contours_data,
            "estratos_geo": estratos_geo_exp,
            "estratos_geotec": estratos_geotec_exp,
            "calicatas_geo": calicatas_geo_exp,
            "calicatas_geotec": calicatas_geotec_exp,
            "deslinde": deslinde_data,
            "edificaciones": edific_data,
            "diccionario": GEO_DICT
        }
        json_data = json.dumps(app_data)

        html_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visor Geológico Dual de Fundaciones</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #121214; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: white; }
        #canvas-container { width: 100vw; height: 100vh; display: block; position: absolute; z-index: 1; }
        #css2d-container { width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; pointer-events: none; z-index: 2; }
        
        #tooltip { position: absolute; background: rgba(25, 25, 28, 0.95); border-left: 4px solid #FFA500; padding: 15px; border-radius: 6px; pointer-events: none; display: none; z-index: 10; box-shadow: 0 8px 25px rgba(0,0,0,0.7); font-size: 13px; min-width: 240px; backdrop-filter: blur(4px); }
        #tooltip h3 { margin: 0 0 10px 0; font-size: 16px; color: #fff; border-bottom: 1px solid #333; padding-bottom: 8px; font-weight: 600; }
        #tooltip .stat { color: #bbb; margin-bottom: 4px; display: flex; justify-content: space-between; font-size: 12px;}
        #tooltip .stat span { color: #fff; font-weight: 500; }
        #tooltip .layer { display: flex; align-items: center; margin-top: 6px; padding: 4px 0; border-bottom: 1px dotted #333; }
        #tooltip .layer:last-child { border-bottom: none; }
        #tooltip .color-box { width: 14px; height: 14px; margin-right: 10px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.2); }
        
        #hud-title { position: absolute; top: 20px; left: 20px; text-shadow: 1px 1px 4px rgba(0,0,0,0.9); pointer-events: none; z-index: 10; }
        #hud-title h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.5px; color: #fff; }
        #hud-title p { margin: 5px 0 0 0; font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
        
        #compass-container { position: absolute; bottom: 30px; left: 30px; width: 60px; height: 60px; z-index: 10; pointer-events: none; }
        #compass { width: 100%; height: 100%; position: relative; transition: transform 0.1s; }
        #compass::before, #compass::after { content: ''; position: absolute; left: 50%; transform: translateX(-50%); border-left: 10px solid transparent; border-right: 10px solid transparent; }
        #compass::before { top: 0; border-bottom: 30px solid #e74c3c; }
        #compass::after { bottom: 0; border-top: 30px solid #ecf0f1; }
        #compass-n { position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-weight: bold; color: #e74c3c; text-shadow: 0 0 5px black; font-size: 16px; }
        
        .coord-label { color: #ccc; font-size: 11px; font-family: monospace; background: rgba(0,0,0,0.6); padding: 2px 5px; border-radius: 3px; border: 1px solid #444; pointer-events: none; user-select: none; white-space: nowrap; transition: opacity 0.3s; }
        .measure-label { color: #fff; font-size: 12px; font-weight: normal; background: rgba(30, 130, 230, 0.9); padding: 5px 8px; border-radius: 4px; pointer-events: none; border: 1px solid #fff; box-shadow: 0 0 10px rgba(30,130,230,0.5); text-align: left;}
        
        .dg.ac { z-index: 20 !important; }
        
        #measure-msg { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(46, 204, 113, 0.9); color: #fff; padding: 10px 20px; border-radius: 20px; font-weight: bold; display: none; z-index: 10; box-shadow: 0 4px 10px rgba(0,0,0,0.5); border: 2px solid #fff; pointer-events: none; font-size: 14px;}
        #measure-msg.cutting { background: rgba(231, 76, 60, 0.9); }
        #measure-msg.drilling { background: rgba(155, 89, 182, 0.9); }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/renderers/CSS2DRenderer.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.9/dat.gui.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tween.js/18.6.4/tween.umd.js"></script>
</head>
<body>

    <div id="hud-title">
        <h1>Análisis de Fundaciones 3D</h1>
        <p>Visor Geológico / Geotécnico</p>
    </div>
    
    <div id="compass-container"><div id="compass-n">N</div><div id="compass"></div></div>
    
    <div id="measure-msg">Modo Activo...</div>
    <div id="tooltip"></div>
    <div id="canvas-container"></div>
    <div id="css2d-container"></div>

    <script>
        const db = __JSON_DATA_PLACEHOLDER__;
        
        function getGeoProp(code, prop) {
            if(db.diccionario && db.diccionario[code]) { return db.diccionario[code][prop]; }
            return code;
        }
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x121214);
        scene.fog = new THREE.FogExp2(0x121214, 0.0003);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 20000);
        const target = new THREE.Vector3(db.center.x, db.center.y, db.center.z);
        camera.position.set(db.center.x + 400, db.center.y - 600, db.center.z + 400);

        const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", alpha: false, preserveDrawingBuffer: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.localClippingEnabled = true;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        const labelRenderer = new THREE.CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('css2d-container').appendChild(labelRenderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.copy(target);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotateSpeed = 1.0;
        controls.maxPolarAngle = Math.PI; 
        controls.screenSpacePanning = true; 
        controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight.position.set(db.center.x - 500, db.center.y - 800, db.bounds.zMax + 1000);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048; dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 10; dirLight.shadow.camera.far = 4000;
        
        const shadowDist = Math.max((db.bounds.xMax - db.bounds.xMin), (db.bounds.yMax - db.bounds.yMin));
        dirLight.shadow.camera.left = -shadowDist; dirLight.shadow.camera.right = shadowDist;
        dirLight.shadow.camera.top = shadowDist; dirLight.shadow.camera.bottom = -shadowDist;
        scene.add(dirLight);

        const clipX = new THREE.Plane(new THREE.Vector3(-1, 0, 0), db.bounds.xMax + 100);
        const clipY = new THREE.Plane(new THREE.Vector3(0, -1, 0), db.bounds.yMax + 100);
        const clipZ = new THREE.Plane(new THREE.Vector3(0, 0, -1), db.bounds.zMax + 100);
        const clipArb = new THREE.Plane(new THREE.Vector3(1, 0, 0), 10000);
        const globalPlanes = [clipX, clipY, clipZ, clipArb];

        const cutPlaneMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false, wireframe: true });
        const cutMeshX = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), cutPlaneMat);
        cutMeshX.rotation.y = Math.PI / 2; cutMeshX.visible = false; scene.add(cutMeshX);
        const cutMeshY = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), cutPlaneMat);
        cutMeshY.rotation.x = Math.PI / 2; cutMeshY.visible = false; scene.add(cutMeshY);
        
        const cutMeshArb = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), new THREE.MeshBasicMaterial({color: 0x00ffff, transparent: true, opacity: 0.15, side: THREE.DoubleSide, depthWrite: false}));
        cutMeshArb.visible = false; scene.add(cutMeshArb);

        const gridGroup = new THREE.Group(); scene.add(gridGroup);
        
        function buildCoordinateGrid() {
            while(gridGroup.children.length > 0) gridGroup.remove(gridGroup.children[0]);
            const w = db.bounds.xMax - db.bounds.xMin; const h = db.bounds.yMax - db.bounds.yMin; const floorZ = db.bounds.zMin - 10;
            const gridHelper = new THREE.GridHelper(Math.max(w, h), 10, 0x555555, 0x333333);
            gridHelper.position.set(db.center.x, db.center.y, floorZ); gridHelper.rotation.x = Math.PI / 2; gridGroup.add(gridHelper);

            const boxGeom = new THREE.BoxGeometry(w, h, db.bounds.zMax - floorZ);
            const boxLines = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), new THREE.LineBasicMaterial({color: 0x666666}));
            boxLines.position.set(db.center.x, db.center.y, floorZ + (db.bounds.zMax - floorZ)/2); gridGroup.add(boxLines);

            function addLabel(text, x, y, z, customStyle=false) {
                const div = document.createElement('div'); div.className = 'coord-label'; 
                if(customStyle) {
                    div.style.background = 'transparent'; div.style.border = 'none'; div.style.fontSize = '14px'; div.style.fontWeight = 'bold'; div.style.textShadow = '1px 1px 2px black';
                }
                div.textContent = text;
                const label = new THREE.CSS2DObject(div); label.position.set(x, y, z); gridGroup.add(label);
            }
            
            // Etiquetas de Coordenadas Originales
            addLabel(`E: ${db.bounds.xMin.toFixed(0)}`, db.bounds.xMin, db.bounds.yMin - 10, floorZ); 
            addLabel(`E: ${db.bounds.xMax.toFixed(0)}`, db.bounds.xMax, db.bounds.yMin - 10, floorZ);
            addLabel(`N: ${db.bounds.yMin.toFixed(0)}`, db.bounds.xMin - 10, db.bounds.yMin, floorZ); 
            addLabel(`N: ${db.bounds.yMax.toFixed(0)}`, db.bounds.xMin - 10, db.bounds.yMax, floorZ);
            addLabel(`Z: ${db.bounds.zMax.toFixed(0)}`, db.bounds.xMin, db.bounds.yMin, db.bounds.zMax);

            // --- NUEVA ESCALA GRÁFICA FÍSICA 3D (POSICIÓN CORREGIDA) ---
            const extent = Math.max(w, h);
            let scaleLen = 10;
            if (extent > 500) scaleLen = 100;
            else if (extent > 100) scaleLen = 50;
            else if (extent > 50) scaleLen = 20;

            const segments = 5;
            const segLen = scaleLen / segments;
            const thick = extent * 0.008; // Ligeramente más gruesa para mejor lectura

            // Calculamos un margen dinámico (8% del tamaño del modelo) hacia el Sur
            const offsetMargin = extent * 0.08; 

            // Posicionamos la escala desplazada hacia el Sur y alineada al Oeste
            const startX = db.bounds.xMin;
            const startY = db.bounds.yMin - offsetMargin;

            for(let i = 0; i < segments; i++) {
                const mat = new THREE.MeshBasicMaterial({ color: i % 2 === 0 ? 0xffffff : 0x000000 });
                const geom = new THREE.BoxGeometry(segLen, thick, thick);
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.set(startX + (i * segLen) + (segLen/2), startY, floorZ);
                gridGroup.add(mesh);
            }
            
            // Etiquetas de la escala adaptadas a la nueva posición
            addLabel('0m', startX, startY - (thick * 2), floorZ, true);
            addLabel(`${scaleLen}m`, startX + scaleLen, startY - (thick * 2), floorZ, true);
        }
        buildCoordinateGrid();

        function createHatchTexture() {
            const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
            const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,64,64); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255, 140, 0, 0.6)';
            ctx.beginPath(); ctx.moveTo(0, 64); ctx.lineTo(64, 0); ctx.stroke();
            const tex = new THREE.CanvasTexture(canvas); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(30, 30); return tex;
        }
        const hatchTexture = createHatchTexture();

        const geoMaterials = {};
        for (const [code, info] of Object.entries(db.diccionario)) {
            geoMaterials[code] = new THREE.MeshStandardMaterial({
                color: new THREE.Color(info.color),
                side: THREE.DoubleSide,
                roughness: 0.7,
                clippingPlanes: globalPlanes,
                transparent: false,
                opacity: 1.0
            });
        }

        const meshes = { topography: null, contours: null, estratosGeo: [], estratosGeotec: [], calicatasGeo: [], calicatasGeotec: [], allInteractable: [] };
        
        function generateElevationColors(geometry) {
            const pos = geometry.attributes.position.array; const colors = new Float32Array(pos.length);
            const zMin = db.bounds.zMin, zRange = db.bounds.zMax - zMin; const colorObj = new THREE.Color();
            for(let i=0; i<pos.length; i+=3) {
                const z = pos[i+2]; const n = Math.max(0, Math.min(1, (z - zMin) / zRange));
                colorObj.setHSL(0.3 - (n * 0.3), 0.8, 0.2 + (n * 0.6));
                colors[i] = colorObj.r; colors[i+1] = colorObj.g; colors[i+2] = colorObj.b;
            }
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        function createMesh(data, colorHex, isTopo=false, geoCode=null) {
            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
            geom.setIndex(data.indices); geom.computeVertexNormals();

            let mat;
            if(isTopo) {
                generateElevationColors(geom);
                mat = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: false, side: THREE.DoubleSide, roughness: 0.9, clippingPlanes: globalPlanes, transparent: true, opacity: 0.15, depthWrite: false });
            } else if (geoCode && geoMaterials[geoCode]) {
                mat = geoMaterials[geoCode];
            } else {
                mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(colorHex), side: THREE.DoubleSide, roughness: 0.7, clippingPlanes: globalPlanes });
            }

            const mesh = new THREE.Mesh(geom, mat);
            if(!isTopo) { mesh.castShadow = true; mesh.receiveShadow = true; }
            scene.add(mesh); meshes.allInteractable.push(mesh); return mesh;
        }

        meshes.topography = createMesh(db.topography, "#88aa88", true);
        const contourGeom = new THREE.BufferGeometry(); contourGeom.setAttribute('position', new THREE.Float32BufferAttribute(db.contours.vertices, 3)); contourGeom.setIndex(db.contours.indices);
        meshes.contours = new THREE.LineSegments(contourGeom, new THREE.LineBasicMaterial({color: 0xdddddd, transparent: true, opacity: 0.6, clippingPlanes: globalPlanes}));
        scene.add(meshes.contours);

        // Render Modelo Geológico
        db.estratos_geo.sort((a, b) => a.order - b.order);
        db.estratos_geo.forEach((layer, index) => {
            const mesh = createMesh(layer, layer.color, false, layer.geo); 
            mesh.userData = layer; mesh.userData.layerIndex = index; 
            meshes.estratosGeo.push(mesh);
        });

        // Render Modelo Geotécnico
        db.estratos_geotec.sort((a, b) => a.order - b.order);
        db.estratos_geotec.forEach((layer, index) => {
            const mesh = createMesh(layer, layer.color, false, layer.geo); 
            mesh.userData = layer; mesh.userData.layerIndex = index; 
            mesh.visible = false; 
            meshes.estratosGeotec.push(mesh);
        });

        const infraGroup = new THREE.Group(); scene.add(infraGroup);
        const lineRadius = (db.bounds.xMax - db.bounds.xMin) * 0.001; 
        
        function createThickLine(verticesArray, colorHex, radius) {
            const points = [];
            for(let i=0; i<verticesArray.length; i+=3) {
                const v = new THREE.Vector3(verticesArray[i], verticesArray[i+1], verticesArray[i+2]);
                if(points.length > 0 && points[points.length-1].distanceTo(v) < 0.01) continue;
                points.push(v);
            }
            if(points.length < 2) return null;
            const path = new THREE.CurvePath();
            for(let i=0; i<points.length-1; i++) path.add(new THREE.LineCurve3(points[i], points[i+1]));
            const geom = new THREE.TubeGeometry(path, points.length * 2, radius, 6, false);
            const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({color: new THREE.Color(colorHex), roughness: 0.5, clippingPlanes: globalPlanes}));
            mesh.castShadow = true; return mesh;
        }
        
        if (db.deslinde) {
            db.deslinde.forEach(item => {
                if(item.type === 'deslinde') {
                    const shape = new THREE.Shape();
                    item.outline.forEach((p, i) => { if(i===0) shape.moveTo(p[0], p[1]); else shape.lineTo(p[0], p[1]); });
                    const mesh = new THREE.Mesh(new THREE.ExtrudeGeometry(shape, { depth: 0.1, bevelEnabled: false }), new THREE.MeshBasicMaterial({map: hatchTexture, transparent: true, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: globalPlanes}));
                    mesh.position.z = item.zBase; infraGroup.add(mesh);
                }
                const lineMesh = createThickLine(item.coords.flat(), 0xff5500, lineRadius * 2.0); 
                if(lineMesh) infraGroup.add(lineMesh);
            });
        }
        
        if (db.edificaciones) {
            db.edificaciones.forEach(item => {
                if (item.type === 'polygon') {
                    const shape = new THREE.Shape();
                    item.outline.forEach((p, i) => { if(i===0) shape.moveTo(p[0], p[1]); else shape.lineTo(p[0], p[1]); });
                    const geom = new THREE.ExtrudeGeometry(shape, { depth: item.height, bevelEnabled: false });
                    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({color: 0x3498db, transparent: true, opacity: 0.7, clippingPlanes: globalPlanes}));
                    mesh.position.z = item.zBase; mesh.castShadow = true; mesh.receiveShadow = true; infraGroup.add(mesh); meshes.allInteractable.push(mesh);
                    const line = new THREE.LineSegments(new THREE.EdgesGeometry(geom), new THREE.LineBasicMaterial({color: 0x2980b9, clippingPlanes: globalPlanes}));
                    line.position.z = item.zBase; infraGroup.add(line);
                }
            });
        }

        const calicataGroupGeo = new THREE.Group(); scene.add(calicataGroupGeo);
        const calicataGroupGeotec = new THREE.Group(); scene.add(calicataGroupGeotec);
        const bhWidth = lineRadius * 8.0;  
        const bhLength = lineRadius * 5.0; 

        function buildCalicatas(bhData, targetGroup, meshList) {
            bhData.forEach(bh => {
                const prof = Math.max(0.01, bh.profTotal); 
                const centerZ = bh.zTop - (prof / 2);
                
                const hitboxGeom = new THREE.BoxGeometry(bhWidth*1.5, bhLength*1.5, prof);
                const hitbox = new THREE.Mesh(hitboxGeom, new THREE.MeshBasicMaterial({visible: false})); 
                hitbox.position.set(bh.x, bh.y, centerZ); 
                hitbox.userData = bh; 
                targetGroup.add(hitbox); 
                meshList.push(hitbox);
                
                bh.segments.forEach(seg => {
                    const height = Math.max(0.01, seg.zTop - seg.zBase); 
                    const boxGeom = new THREE.BoxGeometry(bhWidth, bhLength, height);
                    
                    const mat = geoMaterials[seg.geoCode] || new THREE.MeshStandardMaterial({color: new THREE.Color(seg.color), clippingPlanes: globalPlanes, roughness: 0.5});
                    const box = new THREE.Mesh(boxGeom, mat); 
                    box.position.set(bh.x, bh.y, seg.zTop - (height / 2)); 
                    box.castShadow = true; 
                    
                    const edges = new THREE.EdgesGeometry(boxGeom);
                    const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2, clippingPlanes: globalPlanes });
                    const edgesLine = new THREE.LineSegments(edges, lineMat);
                    box.add(edgesLine);

                    targetGroup.add(box); 
                });
            });
        }

        buildCalicatas(db.calicatas_geo, calicataGroupGeo, meshes.calicatasGeo);
        buildCalicatas(db.calicatas_geotec, calicataGroupGeotec, meshes.calicatasGeotec);
        calicataGroupGeotec.visible = false; 

        meshes.allInteractable.push(cutMeshX, cutMeshY, cutMeshArb);
        const virtualBhGroup = new THREE.Group(); scene.add(virtualBhGroup);

        let interactionMode = 'IDLE'; let actionPoints = []; const actionMarkers = []; let tempObjects = []; let cutHelper = null;

        function resetTools() {
            actionPoints = []; actionMarkers.forEach(m => scene.remove(m)); actionMarkers.length = 0;
            tempObjects.forEach(o => { scene.remove(o); gridGroup.remove(o); }); tempObjects = [];
            if(cutHelper) { scene.remove(cutHelper); cutHelper = null; }
            document.getElementById('measure-msg').style.display = 'none';
        }

        const gui = new dat.GUI({ width: 340 });
        const params = { tipoModelo: 'Geológico', modoMedicion: false, modoCorteLibre: false, modoSondeoV: false, autoRotar: false, verTopografia: true, colorTopografia: 'Cristalino', sombras: true, wireframe: false, opacidadEstratos: 1.0, vistaExplotada: 0, corteX: db.bounds.xMax + 10, invX: false, corteY: db.bounds.yMax + 10, invY: false, corteZ: db.bounds.zMax + 10, invZ: false, mostrarCorteHelper: false };

        const geoKeys = ["SN", "RA", "HA", "HM", "FC", "HG"];
        const geotecKeys = ["CI", "NAC", "NCC"];
        
        // Carpetas dinámicas para la leyenda
        const folderGeo = gui.addFolder('🎨 Leyenda Geológica');
        const folderGeotec = gui.addFolder('🎨 Leyenda Geotécnica');
        
        function addColorToFolder(folder, keys) {
            keys.forEach(code => {
                if(db.diccionario[code]) {
                    const info = db.diccionario[code];
                    const dummy = { [info.name]: info.color };
                    folder.addColor(dummy, info.name).onChange(newColor => {
                        if(geoMaterials[code]) geoMaterials[code].color.set(newColor);
                        db.diccionario[code].color = newColor;
                    });
                }
            });
        }
        
        addColorToFolder(folderGeo, geoKeys);
        addColorToFolder(folderGeotec, geotecKeys);
        
        folderGeo.open();
        folderGeotec.open();
        folderGeotec.domElement.style.display = 'none'; 

        function updateModelVisibility() {
            const isGeo = (params.tipoModelo === 'Geológico');
            meshes.estratosGeo.forEach(m => m.visible = isGeo);
            meshes.estratosGeotec.forEach(m => m.visible = !isGeo);
            calicataGroupGeo.visible = isGeo;
            calicataGroupGeotec.visible = !isGeo;
            
            // Alternar carpetas
            if(isGeo) {
                folderGeo.domElement.style.display = '';
                folderGeotec.domElement.style.display = 'none';
            } else {
                folderGeo.domElement.style.display = 'none';
                folderGeotec.domElement.style.display = '';
            }
        }

        gui.add(params, 'tipoModelo', ['Geológico', 'Geotécnico']).name('🔎 Tipo de Modelo').onChange(updateModelVisibility);

        function animarCamara(posObj, targetObj) {
            new TWEEN.Tween(camera.position).to(posObj, 2000).easing(TWEEN.Easing.Cubic.InOut).start();
            new TWEEN.Tween(controls.target).to(targetObj, 2000).easing(TWEEN.Easing.Cubic.InOut).start();
        }

        const herramientas = {
            capturar: function() { renderer.render(scene, camera); const link = document.createElement('a'); link.download = 'estudio_fundaciones.png'; link.href = renderer.domElement.toDataURL('image/png'); link.click(); },
            mostrarTodo: () => { actualizarVisibilidad(true); updateModelVisibility(); }, 
            ocultarTodo: () => actualizarVisibilidad(false),
            vistaIso: () => animarCamara({x: db.center.x + 400, y: db.center.y - 600, z: db.bounds.zMax + 400}, db.center),
            vistaPlanta: () => animarCamara({x: db.center.x, y: db.center.y, z: db.bounds.zMax + 1500}, db.center),
            vistaPerfilY: () => animarCamara({x: db.center.x, y: db.bounds.yMin - 1200, z: db.center.z}, db.center),
            vistaPerfilX: () => animarCamara({x: db.bounds.xMax + 1200, y: db.center.y, z: db.center.z}, db.center),
            resetCorteLibre: () => { clipArb.constant = 10000; cutMeshArb.visible = false; resetTools(); },
            limpiarSondeos: () => { while(virtualBhGroup.children.length > 0) virtualBhGroup.remove(virtualBhGroup.children[0]); }
        };

        function actualizarVisibilidad(estado) {
            params.verTopografia = estado; meshes.topography.visible = estado; meshes.contours.visible = estado;
            gridGroup.visible = estado; infraGroup.visible = estado; virtualBhGroup.visible = estado;
            meshes.estratosGeo.forEach(m => m.visible = false);
            meshes.estratosGeotec.forEach(m => m.visible = false);
            calicataGroupGeo.visible = false; calicataGroupGeotec.visible = false;
            document.querySelectorAll('.coord-label').forEach(el => el.style.opacity = estado ? '1' : '0');
            for (let i in gui.__folders['Visualización y Estilos'].__controllers) { gui.__folders['Visualización y Estilos'].__controllers[i].updateDisplay(); }
        }

        const f0 = gui.addFolder('👁️ Accesos Rápidos y Cámaras');
        f0.add(herramientas, 'mostrarTodo').name('▶ Mostrar Todo'); f0.add(herramientas, 'ocultarTodo').name('⏸ Ocultar Todo');
        f0.add(herramientas, 'vistaIso').name('🎥 Vista Isométrica'); f0.add(herramientas, 'vistaPlanta').name('🎥 Vista Planta (Cenital)');
        f0.add(herramientas, 'vistaPerfilY').name('🎥 Perfil Sur-Norte'); f0.add(herramientas, 'vistaPerfilX').name('🎥 Perfil Oeste-Este');
        f0.add(params, 'autoRotar').name('Auto-Rotar Escena').onChange(v => controls.autoRotate = v); f0.open();

        const f1 = gui.addFolder('📐 Herramientas Suelos de Fundación');
        f1.add(params, 'modoMedicion').name('📏 Cinta Métrica (Click)').listen().onChange(v => {
            if(v) { interactionMode = 'MEASURE'; params.modoCorteLibre = false; params.modoSondeoV = false; let m = document.getElementById('measure-msg'); m.innerText = "📏 MEDICIÓN EXACTA: Haz clic en dos puntos de la malla o perfiles"; m.className = ""; m.style.display = 'block'; }
            else { interactionMode = 'IDLE'; resetTools(); }
        });
        f1.add(params, 'modoCorteLibre').name('✂️ Perfil Arbitrario (2 Clics)').listen().onChange(v => {
            if(v) { interactionMode = 'CUT'; params.modoMedicion = false; params.modoSondeoV = false; let m = document.getElementById('measure-msg'); m.innerText = "✂️ CORTE LIBRE: Haz clic en el inicio y fin del corte deseado"; m.className = "cutting"; m.style.display = 'block'; }
            else { interactionMode = 'IDLE'; resetTools(); }
        });
        f1.add(params, 'modoSondeoV').name('⛏️ Sondeo Virtual (Click Terreno)').listen().onChange(v => {
            if(v) { interactionMode = 'VIRTUAL_BH'; params.modoMedicion = false; params.modoCorteLibre = false; let m = document.getElementById('measure-msg'); m.innerText = "⛏️ SONDEO VIRTUAL: Haz clic sobre el terreno o edificios para sondear"; m.className = "drilling"; m.style.display = 'block'; }
            else { interactionMode = 'IDLE'; resetTools(); }
        });
        f1.add(herramientas, 'limpiarSondeos').name('🧹 Limpiar Sondeos Virtuales'); f1.add(herramientas, 'resetCorteLibre').name('🔄 Limpiar Corte Libre');
        
        f1.add(params, 'corteX', db.bounds.xMin - 10, db.bounds.xMax + 10).name('Corte Perfil X').onChange(v => { clipX.constant = params.invX ? -v : v; cutMeshX.position.x = v; cutMeshX.visible = params.mostrarCorteHelper; });
        f1.add(params, 'invX').name('🔄 Invertir Corte X').onChange(v => { clipX.normal.set(v ? 1 : -1, 0, 0); clipX.constant = v ? -params.corteX : params.corteX; });
        
        f1.add(params, 'corteY', db.bounds.yMin - 10, db.bounds.yMax + 10).name('Corte Perfil Y').onChange(v => { clipY.constant = params.invY ? -v : v; cutMeshY.position.y = v; cutMeshY.visible = params.mostrarCorteHelper; });
        f1.add(params, 'invY').name('🔄 Invertir Corte Y').onChange(v => { clipY.normal.set(0, v ? 1 : -1, 0); clipY.constant = v ? -params.corteY : params.corteY; });
        
        f1.add(params, 'corteZ', db.bounds.zMin - 10, db.bounds.zMax + 10).name('Excavación Z').onChange(v => { clipZ.constant = params.invZ ? -v : v; });
        f1.add(params, 'mostrarCorteHelper').name('Ver Planos de Corte').onChange(v => { cutMeshX.visible = v && clipX.constant <= db.bounds.xMax; cutMeshY.visible = v && clipY.constant <= db.bounds.yMax; });
        f1.add(params, 'vistaExplotada', 0, 30).name('Vista Explotada (Gap)').onChange(v => { 
            meshes.estratosGeo.forEach(m => m.position.z = -v * m.userData.layerIndex); 
            meshes.estratosGeotec.forEach(m => m.position.z = -v * m.userData.layerIndex); 
            scene.updateMatrixWorld(true); 
        });
        f1.open();

        const f2 = gui.addFolder('Visualización y Estilos');
        f2.add(params, 'verTopografia').name('Topografía y Curvas').onChange(v => { meshes.topography.visible = v; meshes.contours.visible = v; });
        f2.add(params, 'colorTopografia', ['Cristalino', 'Elevación Z']).name('Color Topografía').onChange(v => { meshes.topography.material.vertexColors = (v === 'Elevación Z'); meshes.topography.material.opacity = (v === 'Elevación Z') ? 0.6 : 0.15; meshes.topography.material.needsUpdate = true; });
        
        f2.add(params, 'opacidadEstratos', 0.1, 1.0).name('Opacidad Rocas').onChange(v => { 
            Object.values(geoMaterials).forEach(mat => {
                mat.opacity = v;
                mat.transparent = (v < 1.0);
                mat.needsUpdate = true;
            });
        });
        f2.add(params, 'wireframe').name('Modo Malla (Geometría)').onChange(v => { 
            Object.values(geoMaterials).forEach(mat => { mat.wireframe = v; });
        });
        f2.add(params, 'sombras').name('Sombras Dinámicas').onChange(v => { renderer.shadowMap.enabled = v; scene.traverse(c => { if(c.material) c.material.needsUpdate = true; }); });
        f2.add(herramientas, 'capturar').name('📸 Tomar Captura PNG');

        const raycaster = new THREE.Raycaster(); const mouse = new THREE.Vector2(); const tooltip = document.getElementById('tooltip');
        function getValidIntersections(interactables) {
            return raycaster.intersectObjects(interactables).filter(i => {
                if(!i.object.visible) return false;
                for(let plane of globalPlanes) { if (plane.constant < 9999 && plane.distanceToPoint(i.point) < 0) return false; }
                return true;
            });
        }

        window.addEventListener('mousemove', (e) => {
            if (interactionMode === 'CUT' && actionPoints.length === 1 && cutHelper) {
                mouse.x = (e.clientX / window.innerWidth) * 2 - 1; mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
                raycaster.setFromCamera(mouse, camera); const hits = getValidIntersections(meshes.allInteractable);
                if(hits.length > 0) { const pos = cutHelper.geometry.attributes.position; pos.setXYZ(1, hits[0].point.x, hits[0].point.y, hits[0].point.z); pos.needsUpdate = true; }
            }

            if(interactionMode !== 'IDLE') return; 
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1; mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera); 
            
            const activeBoreholes = (params.tipoModelo === 'Geológico') ? meshes.calicatasGeo : meshes.calicatasGeotec;
            const intersects = getValidIntersections(activeBoreholes);

            if (intersects.length > 0 && ((params.tipoModelo === 'Geológico' && calicataGroupGeo.visible) || (params.tipoModelo === 'Geotécnico' && calicataGroupGeotec.visible))) {
                document.body.style.cursor = 'pointer'; const bh = intersects[0].object.userData;
                let html = `<h3>⛏️ Calicata Histórica: ${bh.id}</h3><div class="stat">Cota Terreno: <span>${bh.zTop.toFixed(2)} m</span></div><div class="stat">Profundidad Total: <span>${bh.profTotal.toFixed(2)} m</span></div><div style="margin-top:10px; margin-bottom:5px; border-bottom:1px solid #333;"></div>`;
                bh.segments.forEach(seg => {
                    const profTecho = bh.zTop - seg.zTop; const profBase = bh.zTop - seg.zBase;
                    const displayColor = db.diccionario[seg.geoCode] ? db.diccionario[seg.geoCode].color : seg.color;
                    html += `<div class="layer"><div class="color-box" style="background:${displayColor}"></div><span><b>${seg.geoName}</b> <br><small style="color:#aaa">Prof: ${profTecho.toFixed(1)}m a ${profBase.toFixed(1)}m</small></span></div>`;
                });
                tooltip.innerHTML = html; tooltip.style.opacity = '1'; tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX + 20 > window.innerWidth - 250 ? e.clientX - 260 : e.clientX + 20) + 'px';
                tooltip.style.top = (e.clientY + 20 > window.innerHeight - 200 ? e.clientY - 210 : e.clientY + 20) + 'px';
            } else { document.body.style.cursor = 'default'; tooltip.style.opacity = '0'; }
        });

        window.addEventListener('click', (e) => {
            if(interactionMode === 'IDLE') return;
            if(e.clientX > window.innerWidth - 350 && e.clientY < 500) return;
            
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1; mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = getValidIntersections(meshes.allInteractable);
            if(intersects.length === 0) return; const pt = intersects[0].point;

            if(interactionMode === 'CUT') {
                actionPoints.push(pt);
                const marker = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 16), new THREE.MeshBasicMaterial({color: 0x00ffff}));
                marker.position.copy(pt); scene.add(marker); actionMarkers.push(marker);
                
                if(actionPoints.length === 1) {
                    cutHelper = new THREE.Line(new THREE.BufferGeometry().setFromPoints([pt, pt]), new THREE.LineBasicMaterial({color: 0x00ffff, linewidth: 3, transparent:true, opacity:0.8})); scene.add(cutHelper);
                }
                
                if(actionPoints.length === 2) {
                    const p1 = actionPoints[0]; const p2 = actionPoints[1];
                    const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
                    let normal = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0,0,1)).normalize();
                    
                    const toCam = new THREE.Vector3().subVectors(camera.position, p1);
                    if (normal.dot(toCam) > 0) normal.negate();
                    
                    clipArb.setFromNormalAndCoplanarPoint(normal, p1);
                    cutMeshArb.position.copy(p1); cutMeshArb.lookAt(new THREE.Vector3().addVectors(p1, normal)); cutMeshArb.visible = params.mostrarCorteHelper;

                    interactionMode = 'MEASURE'; params.modoCorteLibre = false; params.modoMedicion = true;
                    document.getElementById('measure-msg').innerText = "📏 El corte expuso el subsuelo hacia la cámara. Haz clic en la pared de roca para medir espesores."; resetTools();
                }
            } 
            else if (interactionMode === 'MEASURE') {
                const marker = new THREE.Mesh(new THREE.SphereGeometry(2, 16, 16), new THREE.MeshBasicMaterial({color: 0xe74c3c}));
                marker.position.copy(pt); scene.add(marker); actionMarkers.push(marker); actionPoints.push(pt);

                if(actionPoints.length === 2) {
                    measureLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(actionPoints), new THREE.LineBasicMaterial({color: 0xe74c3c, linewidth: 3, depthTest: false}));
                    scene.add(measureLine); tempObjects.push(measureLine);

                    const p1 = actionPoints[0]; const p2 = actionPoints[1];
                    const dist = p1.distanceTo(p2); const dz = Math.abs(p1.z - p2.z); const dxy = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                    const midPt = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                    
                    const div = document.createElement('div'); div.className = 'measure-label';
                    div.innerHTML = `Dist. Directa: <b>${dist.toFixed(2)} m</b><br><span style="color:#aaf">Dist. Horiz (ΔXY): ${dxy.toFixed(2)} m</span><br><span style="color:#faa">Profundidad (ΔZ): ${dz.toFixed(2)} m</span>`;
                    measureLabel = new THREE.CSS2DObject(div); measureLabel.position.copy(midPt); gridGroup.add(measureLabel); tempObjects.push(measureLabel);
                    actionPoints = []; 
                }
            }
            else if (interactionMode === 'VIRTUAL_BH') {
                const vRay = new THREE.Raycaster(new THREE.Vector3(pt.x, pt.y, db.bounds.zMax + 500), new THREE.Vector3(0,0,-1));
                const estratosHits = [];
                
                const activeEstratos = (params.tipoModelo === 'Geológico') ? meshes.estratosGeo : meshes.estratosGeotec;
                
                activeEstratos.forEach(mesh => {
                    if(!mesh.visible) return;
                    let hits = vRay.intersectObject(mesh);
                    if(hits.length > 0) {
                        hits.sort((a,b) => b.point.z - a.point.z);
                        const zTop = hits[0].point.z; const zBase = hits[hits.length-1].point.z;
                        if (zTop - zBase >= 0.01) { 
                            estratosHits.push({
                                geoCode: mesh.userData.geo,
                                geoName: getGeoProp(mesh.userData.geo, "name"),
                                zTop: zTop, zBase: zBase
                            });
                        }
                    }
                });
                
                if(estratosHits.length === 0) return;
                estratosHits.sort((a,b) => b.zTop - a.zTop);
                const surfaceZ = pt.z; 

                const vBhRadius = lineRadius * 2.5;
                estratosHits.forEach(seg => {
                    const h = seg.zTop - seg.zBase;
                    const mat = geoMaterials[seg.geoCode] || new THREE.MeshStandardMaterial({color: 0xffffff});
                    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(vBhRadius, vBhRadius, h, 16).rotateX(Math.PI/2), mat); 
                    cyl.position.set(pt.x, pt.y, seg.zTop - h/2); virtualBhGroup.add(cyl);
                });
                
                const div = document.createElement('div'); div.className = 'measure-label'; div.style.background = 'rgba(155, 89, 182, 0.95)';
                let html = `<div style="border-bottom:1px solid #fff; margin-bottom:5px; padding-bottom:3px;"><b>📍 SONDEO VIRTUAL</b><br><small>E:${pt.x.toFixed(0)} N:${pt.y.toFixed(0)} | Cota Collar: ${surfaceZ.toFixed(1)}m</small></div>`;
                
                let accumulatedDepth = 0.0;
                estratosHits.forEach(seg => {
                    const thickness = seg.zTop - seg.zBase;
                    const profTop = accumulatedDepth;
                    const profBase = accumulatedDepth + thickness;
                    accumulatedDepth = profBase;
                    const displayColor = db.diccionario[seg.geoCode] ? db.diccionario[seg.geoCode].color : "#fff";
                    
                    html += `<div style="margin-top:3px;"><span style="display:inline-block; width:10px; height:10px; background:${displayColor}; margin-right:5px; border-radius:2px;"></span>${seg.geoName} (${profTop.toFixed(2)}m a ${profBase.toFixed(2)}m)</div>`;
                });
                
                div.innerHTML = html;
                const vLabel = new THREE.CSS2DObject(div); vLabel.position.set(pt.x, pt.y, surfaceZ + 15); virtualBhGroup.add(vLabel);
            }
        });

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight); labelRenderer.setSize(window.innerWidth, window.innerHeight);
        });

        const compassElem = document.getElementById('compass');
        function animate() {
            requestAnimationFrame(animate);
            TWEEN.update(); controls.update();
            const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
            compassElem.style.transform = `rotate(${Math.atan2(dir.x, dir.y)}rad)`;
            renderer.render(scene, camera); labelRenderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>"""

        html_content = html_template.replace("__JSON_DATA_PLACEHOLDER__", json_data)

        with open(EXPORT_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        log(f"¡Éxito! Aplicación generada con capas completas: {EXPORT_HTML}")
        abs_path = os.path.abspath(EXPORT_HTML)
        webbrowser.open(f"file://{abs_path}")

    except Exception:
        log_section("ERROR FATAL")
        traceback.print_exc()

if __name__ == "__main__":
    main()