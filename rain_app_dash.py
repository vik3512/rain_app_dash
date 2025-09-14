#!/usr/bin/env python
# -*- coding: utf-8 -*-
# App Version: 33.1-stable-final
# 規格：搜尋/定位＝移動畫面→面積掃描→ moveend 3秒內用真實 bounds 再掃一次
# 視覺：搜尋/定位加 Marker；區域掃描不加 Marker；不畫掃描圈
# 競態：任務計數 started/finished 推導 busy
# 快取：HOURLY_CACHE = TTLCache(maxsize=8000, ttl=3600)
# 穩定：初始化 i18n 有預設，viewport/center 型別兼容，dispatcher try/except，無 key 參數

import os, time, math, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import Dash, html, dcc, Input, Output, State, no_update
from cachetools import cached, TTLCache

# EventListener 兼容
try:
    from dash_extensions import EventListener
except Exception:
    from dash_extensions.enrich import EventListener

# ctx 兼容
try:
    from dash import ctx
except Exception:
    from dash import callback_context as ctx  # type: ignore

def _triggered_id():
    try:
        tid = ctx.triggered_id
        if tid is not None: return tid
    except Exception:
        pass
    try:
        trig = getattr(ctx, "triggered", None)
        if trig:
            prop_id = trig[0].get("prop_id", "")
            return prop_id.split(".")[0] if prop_id else None
    except Exception:
        pass
    return None

# --------------------------- 常數 / I18N ---------------------------
DEFAULT_TW_CENTER  = [23.9738, 120.9820]
DEFAULT_TW_ZOOM    = 7
SEARCHED_ZOOM      = 13

BASEMAPS = {
    "low": {"url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "attr": "&copy; OpenStreetMap & CARTO"},
    "osm": {"url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attr": "&copy; OpenStreetMap contributors"},
}

CIRCLE_LIGHT_BLUE = "#96b7ff"
CIRCLE_DEEP_BLUE  = "#3a66e6"
CIRCLE_PURPLE     = "#6a42d9"

CURRENT_MM_MIN_MM = 1.0
SUSPECT_MM_MIN_MM = 0.2
ALERT_MM_MIN_MM   = 0.2
THUNDER_MIN, THUNDER_MAX = 95, 99

INTERNATIONAL_KEYWORDS = ["澳洲","美國","日本","英國","法國","德國","中國","香港","澳門","韓國","加拿大","紐西蘭","泰國"]
LANG_MAP = {"zh":"zh-TW","en":"en","ja":"ja"}
I18N = {
    "zh": {"search_here":"在此區域搜尋","searching":"搜尋中...","err_not_found":"找不到您輸入的地點，請換個關鍵字再試一次。",
           "data_updated":"資料更新於","maybe_rain":"可能降雨","map_center":"地圖中心點","basemap":"底圖","low":"低飽和","osm":"原始",
           "geo_denied":"定位權限被拒絕，請在瀏覽器或系統設定開啟定位權限。","geo_unavailable":"定位服務暫時不可用，請確認網路/訊號或稍後重試。",
           "geo_timeout":"定位逾時，請移動到開闊處或再試一次。","geo_error":"定位失敗，請稍後再試。"},
    "en": {"search_here":"Search this area","searching":"Searching...","err_not_found":"Can't find that place. Try another keyword.",
           "data_updated":"Updated at","maybe_rain":"Possible rain","map_center":"Map center","basemap":"Basemap","low":"Low-sat","osm":"Original",
           "geo_denied":"Location permission denied. Please enable location in browser/system settings.",
           "geo_unavailable":"Position unavailable. Check network/signal and try again.",
           "geo_timeout":"Location request timed out. Move to an open area and retry.","geo_error":"Location failed. Please try again."},
    "ja": {"search_here":"このエリアを検索","searching":"検索中...","err_not_found":"場所が見つかりません。別のキーワードでお試しください。",
           "data_updated":"更新時刻","maybe_rain":"降雨の可能性","map_center":"地図の中心","basemap":"ベースマップ","low":"低彩度","osm":"オリジナル",
           "geo_denied":"位置情報の許可が拒否されました。ブラウザ/OSで位置情報を有効にしてください。",
           "geo_unavailable":"位置情報を取得できません。ネットワーク/電波状況を確認してください。",
           "geo_timeout":"位置情報の取得がタイムアウトしました。開けた場所で再試行してください。",
           "geo_error":"位置情報の取得に失敗しました。もう一度お試しください。"}
}

# --------------------------- 設定 ---------------------------
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3,
                  status_forcelist=[429,500,502,503,504], allowed_methods={"GET"})
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update({"User-Agent": "rain-map-dash/33.1-final"})
    return s

SESSION = _make_session()
api_cache = TTLCache(maxsize=256, ttl=300)
logging.basicConfig(level=logging.INFO)

# 小時級 API 結果快取
HOURLY_CACHE = TTLCache(maxsize=8000, ttl=3600)
def _current_hour_key_utc(): return datetime.utcnow().strftime("%Y-%m-%dT%H")

# --------------------------- Utils ---------------------------
def _normalize_center(center):
    try:
        if isinstance(center, dict):
            lat = center.get("lat")
            lon = center.get("lng", center.get("lon"))
        elif isinstance(center, (list, tuple)) and len(center) >= 2:
            lat, lon = center[0], center[1]
        else:
            return None, None
        lat = float(lat); lon = float(lon)
        if abs(lat) > 90 or abs(lon) > 180: return None, None
        return lat, lon
    except Exception:
        return None, None

def _vp_center(viewport):
    if viewport is None: return (None, None)
    if isinstance(viewport, dict):  return _normalize_center(viewport.get("center"))
    if isinstance(viewport, (list, tuple)): return _normalize_center(viewport)
    return (None, None)

def _any_center(obj):
    if obj is None: return (None, None)
    if isinstance(obj, (list, tuple, dict)): return _normalize_center(obj)
    return (None, None)

def _safe_viewport(lat, lon, zoom):
    try:   return dict(center=[float(lat), float(lon)], zoom=int(zoom))
    except Exception: return None

def _quantize(v, q=0.05): return round(round(float(v)/q)*q, 5)
def _quantize_pair(lat, lon, q=0.05): return (_quantize(lat, q), _quantize(lon, q))

def _bounds_from_center_zoom(lat: float, lon: float, zoom: int | float):
    z = int(zoom or 13)
    if z >= 14: e = 0.10
    elif z >= 13: e = 0.15
    elif z >= 12: e = 0.25
    elif z >= 11: e = 0.40
    elif z >= 10: e = 0.60
    else: e = 0.90
    return [[lat - e, lon - e], [lat + e, lon + e]]

# --------------------------- 地名解析 ---------------------------
def _expand_candidates(q: str):
    q = (q or "").strip()
    base = [q] if q else []
    if q and (len(q) <= 4 or q.isdigit()): base += [f"{q} 台北", f"台北 {q}"]
    if q == "101": base = ["台北101", "Taipei 101"] + base
    return list(dict.fromkeys(base))

@cached(api_cache)
def _geocode_google(q: str, scope: str = "tw", lang: str = "zh-TW"):
    params = {"address": q, "key": GOOGLE_MAPS_API_KEY, "language": lang}
    if scope == "tw": params.update({"region": "tw", "components": "country:TW"})
    try:
        r = SESSION.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=10).json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR"
    if r.get("status") == "OK" and r.get("results"):
        top = r["results"][0]; loc = top["geometry"]["location"]
        return top.get("formatted_address", q), (loc.get("lat"), loc.get("lng")), None
    return None, (None, None), r.get("status") or "NO_RESULT"

@cached(api_cache)
def _places_findplace(q: str, lang: str = "zh-TW"):
    if not GOOGLE_MAPS_API_KEY: return None, (None, None), "NO_KEY"
    params = {"input": q, "inputtype": "textquery",
              "fields":"geometry,formatted_address,name","language":lang,"key":GOOGLE_MAPS_API_KEY}
    try:
        r = SESSION.get("https://maps.googleapis.com/maps/api/place/findplacefromtext/json", params=params, timeout=10).json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR"
    c = r.get("candidates") or []
    if r.get("status") == "OK" and c:
        loc = c[0]["geometry"]["location"]
        addr = c[0].get("formatted_address") or c[0].get("name") or q
        return addr, (float(loc.get("lat")), float(loc.get("lng"))), None
    return None, (None, None), r.get("status") or "NO_RESULT"

@cached(api_cache)
def _geocode_nominatim(q: str, lang: str = "zh-TW", tw_only: bool = True):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "addressdetails": 1, "accept-language": lang, "limit": 1}
    if tw_only: params["countrycodes"] = "tw"
    try:
        r = SESSION.get(url, params=params, timeout=10).json()
    except Exception:
        return None, (None, None), "REQUEST_ERROR"
    if isinstance(r, list) and r:
        top = r[0]
        return top.get("display_name") or q, (float(top["lat"]), float(top["lon"])), None
    return None, (None, None), "NO_RESULT"

def smart_geocode(q: str, lang_code: str):
    lang = LANG_MAP.get(lang_code, "zh-TW")
    if not q: return {}
    is_intl = any(k in q for k in INTERNATIONAL_KEYWORDS)
    if not is_intl:
        for cand in _expand_candidates(q):
            addr,(lat,lon),_ = _geocode_google(cand, scope="tw", lang=lang)
            if lat and lon: return {"address":addr,"lat":lat,"lon":lon}
        for cand in _expand_candidates(q):
            addr,(lat,lon),_ = _places_findplace(cand, lang=lang)
            if lat and lon: return {"address":addr,"lat":lat,"lon":lon}
    addr,(lat,lon),_ = _geocode_google(q, scope="global", lang=lang)
    if lat and lon: return {"address":addr,"lat":lat,"lon":lon}
    addr,(lat,lon),_ = _geocode_nominatim(q, lang=lang, tw_only=False)
    if lat and lon: return {"address":addr,"lat":lat,"lon":lon}
    return {}

# --------------------------- 反向地理 ---------------------------
@cached(api_cache)
def _reverse_geocode_google(lat: float, lon: float, lang: str = "zh-TW", prefer_area: bool = False):
    if not GOOGLE_MAPS_API_KEY: return None, "NO_KEY"
    params = {"latlng": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY, "language": lang}
    if prefer_area:
        params["result_type"] = ("neighborhood|sublocality|locality|postal_town|"
            "administrative_area_level_3|administrative_area_level_2|administrative_area_level_1")
    try:
        r = SESSION.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=10).json()
    except Exception:
        return None, "REQUEST_ERROR"
    results = r.get("results") or []
    if r.get("status") == "OK" and results:
        if not prefer_area:
            for item in results:
                types = set(item.get("types") or [])
                if not ({"establishment","point_of_interest","premise"} & types):
                    return item.get("formatted_address"), None
        return results[0].get("formatted_address"), None
    return None, r.get("status") or "NO_RESULT"

def reverse_geocode(lat: float, lon: float, lang_code: str, prefer_area: bool = False):
    lang = LANG_MAP.get(lang_code, "zh-TW")
    addr, _ = _reverse_geocode_google(lat, lon, lang=lang, prefer_area=prefer_area)
    if addr: return addr
    try:
        r = SESSION.get("https://nominatim.openstreetmap.org/reverse",
                        params={"lat": lat, "lon": lon, "format": "json",
                                "accept-language": lang, "zoom": (14 if prefer_area else 18)},
                        timeout=10).json()
        return r.get("display_name")
    except Exception:
        return None

# --------------------------- 天氣資料 ---------------------------
def _get_step_for_zoom(zoom: int | float) -> float:
    z = float(zoom or 0)
    if z >= 12: return 0.03
    if z >= 11: return 0.05
    if z >= 10: return 0.08
    if z >= 9:  return 0.10
    if z >= 8:  return 0.15
    return 0.20

def _cap_step_for_points(south, north, west, east, step, max_points=400):
    lat_span = max(0.0, float(north) - float(south))
    lon_span = max(0.0, float(east)  - float(west))
    if step <= 0: step = 0.2
    while True:
        n_lat = int(lat_span/step) + 1
        n_lon = int(lon_span/step) + 1
        if n_lat * n_lon <= max_points: return step
        step *= 1.5

@cached(api_cache)
def _om_hourly_now(lat: float, lon: float):
    try:
        r = SESSION.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude": lat, "longitude": lon,
                                "hourly":"precipitation,weather_code",
                                "forecast_days":1, "timezone":"auto"},
                        timeout=10).json()
        h=r.get("hourly",{}); t=h.get("time",[])
        now=datetime.now(timezone(timedelta(seconds=r.get("utc_offset_seconds",0))))
        k=now.strftime("%Y-%m-%dT%H:00")
        idx=t.index(k) if k in t else 0
        prec=h.get("precipitation",[]) or [0]
        mm_now = float(prec[idx]) if idx<len(prec) else 0.0
        mm_next= float(prec[idx+1]) if (idx+1)<len(prec) else 0.0
        codes=h.get("weather_code",[]) or [0]
        code_om=int(codes[idx]) if idx<len(codes) else 0
        return mm_now, None, code_om, mm_next
    except Exception:
        return 0.0, None, 0, 0.0

def _get_hourly_with_quantized_cache(lat: float, lon: float, q=0.05):
    qlat, qlon = _quantize_pair(lat, lon, q=q)
    hour_key = _current_hour_key_utc()
    memo_key = (qlat, qlon, hour_key)
    if memo_key in HOURLY_CACHE:
        return HOURLY_CACHE[memo_key]
    mm_now, _, code_om, mm_next = _om_hourly_now(qlat, qlon)
    HOURLY_CACHE[memo_key] = (mm_now, code_om, mm_next)
    return (mm_now, code_om, mm_next)

def _is_current_rainy(mm_now: float, code_om: int) -> bool:
    return (mm_now >= CURRENT_MM_MIN_MM) or (THUNDER_MIN <= code_om <= THUNDER_MAX)

def _is_suspect(mm_now: float, code_om: int) -> bool:
    return (mm_now >= SUSPECT_MM_MIN_MM) or (THUNDER_MIN <= code_om <= THUNDER_MAX)

def _anchors_from_bounds(bounds):
    (south, west), (north, east) = bounds
    cy = (south + north) / 2.0
    cx = (west  + east)  / 2.0
    return [(cy, cx), (south, west), (south, east), (north, west), (north, east)]

def _coarse_centers(bounds, n=3):
    (south, west), (north, east) = bounds
    lat_step = (north - south) / n
    lon_step = (east  - west)  / n
    centers = []
    cells   = []
    for i in range(n):
        for j in range(n):
            clat = south + (i + 0.5) * lat_step
            clon = west  + (j + 0.5) * lon_step
            centers.append((clat, clon))
            cells.append(((south + i*lat_step, west + j*lon_step),
                          (south + (i+1)*lat_step, west + (j+1)*lat_step)))
    return centers, cells

def _gen_fine_points_for_cell(cell_bounds, step):
    (south, west), (north, east) = cell_bounds
    pts = []
    lat_r = south
    while lat_r <= north + 1e-9:
        lon_r = west
        while lon_r <= east + 1e-9:
            pts.append((round(lat_r,5), round(lon_r,5)))
            lon_r += step
        lat_r += step
    return pts

def get_weather_data_for_bounds(bounds, zoom):
    try:
        (south_lat, west_lon), (north_lat, east_lon) = bounds
        south_lat = float(south_lat); west_lon = float(west_lon)
        north_lat = float(north_lat); east_lon = float(east_lon)
    except Exception:
        return [], _get_step_for_zoom(zoom or 8)

    fine_step = _cap_step_for_points(south_lat, north_lat, west_lon, east_lon,
                                     _get_step_for_zoom(zoom), max_points=400)

    # 早退：中心+四角
    anchor_pts = _anchors_from_bounds(((south_lat, west_lon), (north_lat, east_lon)))
    all_dry = True
    try:
        for (ay, ax) in anchor_pts:
            mm, code, _ = _get_hourly_with_quantized_cache(ay, ax)
            if _is_suspect(mm, code): all_dry = False; break
        if all_dry: return [], fine_step
    except Exception as e:
        logging.error(f"Anchor check error: {e}")

    # 粗格掃描 → 細掃
    centers, cells = _coarse_centers(((south_lat, west_lon), (north_lat, east_lon)), n=3)
    suspect_cells_idx = []
    try:
        for idx, (cy, cx) in enumerate(centers):
            mm, code, _ = _get_hourly_with_quantized_cache(cy, cx)
            if _is_suspect(mm, code): suspect_cells_idx.append(idx)
    except Exception as e:
        logging.error(f"Coarse scan error: {e}")

    uniq_points = set()
    for idx in suspect_cells_idx:
        cell = cells[idx]
        for (py, px) in _gen_fine_points_for_cell(cell, fine_step):
            qy, qx = _quantize_pair(py, px, q=0.05)
            uniq_points.add((qy, qx))

    if not uniq_points and suspect_cells_idx:
        for idx in suspect_cells_idx:
            cy, cx = centers[idx]
            qy, qx = _quantize_pair(cy, cx, q=0.05)
            uniq_points.add((qy, qx))

    if not uniq_points:
        return [], fine_step

    n_tasks = len(uniq_points)
    pool_sz = min(12, max(4, math.ceil(n_tasks / 8)))

    points_with_rain = []
    def work(pt):
        glat, glon = pt
        mm_now, code_om, _ = _get_hourly_with_quantized_cache(glat, glon)
        if _is_current_rainy(mm_now, code_om):
            return (glat, glon, mm_now)
        return None

    with ThreadPoolExecutor(pool_sz) as ex:
        futures = [ex.submit(work, p) for p in uniq_points]
        for f in as_completed(futures):
            try:
                res = f.result()
                if res: points_with_rain.append(res)
            except Exception as e:
                logging.error(f"Weather point fetch error: {e}")

    return points_with_rain, fine_step

def get_weather_data_around(lat: float, lon: float):
    extent = 0.15
    return get_weather_data_for_bounds([[lat-extent, lon-extent],[lat+extent, lon+extent]], zoom=10)

# --------------------------- 視覺化（雨圈） ---------------------------
def _circle_style_for_basemap(basemap_mode: str):
    if basemap_mode == "osm":
        return dict(radius_boost=1.45, fill_base=0.34, fill_gain=0.40, stroke_w=0.9, stroke_op=0.85)
    return dict(radius_boost=1.00, fill_base=0.22, fill_gain=0.26, stroke_w=0.6, stroke_op=0.75)

def _color_by_mm(mm: float):
    if mm < 2.0:  return CIRCLE_LIGHT_BLUE
    if mm < 5.0:  return CIRCLE_DEEP_BLUE
    return CIRCLE_PURPLE

def build_rain_circles(wx_points, step_degree: float, basemap_mode: str):
    if not wx_points: return []
    s = _circle_style_for_basemap(basemap_mode)
    base_radius_m = (float(step_degree) * 111000) / 2 * 0.9
    layers=[]
    for lat, lon, mm in wx_points:
        intensity    = max(0.0, min(1.0, float(mm) / 6.0))
        color        = _color_by_mm(mm)
        radius_m     = base_radius_m * (0.9 + 0.9 * intensity) * s["radius_boost"]
        fill_opacity = min(0.72, s["fill_base"] + s["fill_gain"] * intensity)
        layers.append(dl.Circle(center=(lat, lon), radius=radius_m,
                                color=color, opacity=s["stroke_op"], weight=s["stroke_w"],
                                fill=True, fillColor=color, fillOpacity=fill_opacity))
    return layers

# --------------------------- App ---------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "即時雨區地圖"

app.index_string = """
<!DOCTYPE html><html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  html,body,#_dash-app-content{height:100%;margin:0;overflow:hidden}
  .leaflet-control-container .leaflet-bottom.leaflet-left{margin-bottom:.5rem;margin-left:.5rem}

  /* ====== 新增：搜尋此區域按鈕的響應式定位（避免被面板遮住） ====== */
  .search-area-btn{
    position:absolute;
    left:50%;
    transform:translateX(-50%);
    top:calc(max(1rem, env(safe-area-inset-top)) + .5rem); /* 預設：頂部置中 */
    z-index:1200; /* 高於面板(1002)與 dcc.Loading */
  }
  /* 手機寬度：移到底部置中，避開 Safari 底部工具列與 safe area */
  @media (max-width: 480px){
    .search-area-btn{
      top:auto;
      bottom:calc(max(1rem, env(safe-area-inset-bottom)) + 3.25rem);
    }
  }
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

loc_events = [
    {"event":"locationfound","props":["type","latlng"]},
    {"event":"locationerror","props":["type","message","code"]},
    {"event":"moveend","props":["type"]}
]

basemap_default = dl.LayerGroup(
    id="basemap-layer",
    children=[dl.TileLayer(url=BASEMAPS["low"]["url"], attribution=BASEMAPS["low"]["attr"])]
)

map_core = dl.Map(
    id="map", center=DEFAULT_TW_CENTER, zoom=DEFAULT_TW_ZOOM,
    children=[
        basemap_default,
        dl.LayerGroup(id="weather-layer"),
        dl.ZoomControl(position='bottomleft'),
        dl.LocateControl(
            position="bottomleft", setView=True, flyTo=True,
            keepCurrentZoomLevel=False, drawMarker=False, drawCircle=True,
            showPopup=False, metric=True,
        ),
    ],
    style={'width':'100vw','height':'100vh','zIndex':0}, zoomControl=False
)
map_with_listener = EventListener(id="loc_listener", events=loc_events, children=map_core)

map_container = html.Div([
    dcc.Loading(children=[map_with_listener]),
    # ====== 變更點：按鈕改用 className 控制定位（移除 inline top/left/transform/zIndex） ======
    dbc.Button(
        id="search-area-button", n_clicks=0, color="light",
        className="search-area-btn shadow-sm",
        children=I18N["zh"]["search_here"],
        style={
            "backgroundColor":"rgba(255,255,255,0.95)","border":"1px solid #d0d7de",
            "color":"#334155","padding":"0.38rem 0.8rem","fontSize":"0.95rem",
            "minWidth":"160px","whiteSpace":"nowrap","borderRadius":"0.5rem"
        }
    )
], style={"position":"relative","width":"100vw","height":"100vh"})

control_panel = dbc.Card([
    html.H3("🗺️ 即時雨區地圖", className="mb-3"),
    dbc.InputGroup([
        dcc.Input(id="search-input", placeholder="輸入地點", type="text", n_submit=0,
                  style={"flex":"1 1 auto","minWidth":"0","height":"38px","padding":"0 .5rem",
                         "border":"1px solid #ced4da","borderRight":"0",
                         "borderRadius":"0.375rem 0 0 0.375rem","outline":"none"}),
        dbc.Button("🔍", id="search-button", n_clicks=0, color="primary",
                   className="shadow-sm",
                   style={"height":"38px","borderRadius":"0 0.375rem 0.375rem 0","border":"1px solid #0d6efd"}),
    ], className="w-100"),
    dbc.Row([
        dbc.Col(html.Small(id="basemap-label", children=I18N["zh"]["basemap"]), width="auto"),
        dbc.Col(dbc.RadioItems(id="basemap-mode",
               options=[{"label": I18N["zh"]["low"], "value":"low"},
                        {"label": I18N["zh"]["osm"], "value":"osm"}],
               value="low", inline=True))
    ], className="mt-2"),
    dbc.Row([
        dbc.Col(dbc.RadioItems(id="lang-mode",
               options=[{"label":"中文","value":"zh"},{"label":"English","value":"en"},{"label":"日本語","value":"ja"}],
               value="zh", inline=True))
    ], className="mt-1"),
    html.Div(id="status-message", className="mt-2"),
    html.Div(id="next-hour-alert", className="mt-2"),
], style={'position':'fixed','top':'max(1rem, env(safe-area-inset-top))','left':'max(1rem, env(safe-area-inset-left))',
          'width':'clamp(300px, 90vw, 380px)','zIndex':1002,'padding':'1rem'})

error_toast = dbc.Toast(id="error-toast", header="", is_open=False, dismissable=True, duration=5000,
                        style={"position":"fixed","top":"1rem","right":"1rem","zIndex":1100})

app.layout = html.Div([
    map_container, control_panel, error_toast,
    dcc.Store(id='action-store'),
    dcc.Store(id='ui-store', data={"rescan_pending": False, "rescan_until": 0.0}),
    dcc.Store(id='task-counters', data={"started": 0, "finished": 0}),
    dcc.Store(id='map-center'),
    dcc.Store(id='location-store'),
])

# --------------------------- Callbacks ---------------------------
@app.callback(
    Output("basemap-label","children"),
    Output("basemap-mode","options"),
    Input("lang-mode","value")
)
def i18n_ui(lang_code):
    t = I18N.get(lang_code) or I18N["zh"]
    return (t.get("basemap","Basemap"),
            [{"label":t.get("low","Low-sat"),"value":"low"},
             {"label":t.get("osm","Original"),"value":"osm"}])

@app.callback(Output('basemap-layer','children'), Input('basemap-mode','value'))
def update_basemap(mode):
    cfg = BASEMAPS.get(mode, BASEMAPS["low"])
    return [dl.TileLayer(url=cfg["url"], attribution=cfg["attr"])]

@app.callback(
    Output("map-center","data"),
    Input("loc_listener","n_events"),
    State("loc_listener","event"),
    State("map","viewport"),
    prevent_initial_call=True
)
def keep_map_center(_n, event, viewport):
    if not event or event.get("type")!="moveend": return no_update
    if viewport:
        lat, lon = _vp_center(viewport)
        if lat is not None and lon is not None:
            return {"lat":lat, "lon":lon}
    return no_update

@app.callback(
    Output("search-area-button","children"),
    Output("search-area-button","disabled"),
    Input("lang-mode","value"),
    Input("task-counters","data")
)
def search_button_ui(lang_code, counters):
    t = I18N.get(lang_code, I18N["zh"])
    counters = counters or {}
    busy = int(counters.get("started",0)) > int(counters.get("finished",0))
    return (t.get("searching","Searching...") if busy else t.get("search_here","Search this area"),
            busy)

def _inc_started(c):
    c = c or {}
    return {"started": int(c.get("started",0)) + 1, "finished": int(c.get("finished",0))}
def _inc_finished(c):
    c = c or {}
    return {"started": int(c.get("started",0)), "finished": int(c.get("finished",0)) + 1}

# 事件分派（try/except 保底）
@app.callback(
    Output('action-store','data'),
    Output('ui-store','data', allow_duplicate=True),
    Output('search-input','value', allow_duplicate=True),
    Output('task-counters','data', allow_duplicate=True),
    Input('search-button','n_clicks'),
    Input('search-input','n_submit'),
    Input('loc_listener','n_events'),
    Input('search-area-button','n_clicks'),
    State('loc_listener','event'),
    State('search-input','value'),
    State('lang-mode','value'),
    State('map-center','data'),
    State('map','viewport'),
    State('map','center'),
    State('ui-store','data'),
    State('task-counters','data'),
    prevent_initial_call=True
)
def dispatcher(n_click, n_submit, _n_events, n_area, event, query, lang_code,
               center_data, viewport, map_center, ui, counters):
    now = time.time()
    lang_code = lang_code or "zh"
    trig = _triggered_id()
    try:
        # 1) 文字搜尋
        if trig in ("search-button", "search-input"):
            q = (query or "").strip()
            if not q:
                return no_update, no_update, no_update, no_update
            deadline = now + 3.0
            return ({"type":"search","q":q,"lang":lang_code,"nonce":now},
                    {"rescan_pending": True, "rescan_until": deadline},
                    "",
                    _inc_started(counters))

        # 2) 定位成功
        if trig == "loc_listener" and event and event.get("type")=="locationfound":
            latlng = event.get("latlng") or {}
            lat, lon = latlng.get("lat"), latlng.get("lng")
            if lat is None or lon is None:
                return ({"type":"geo_error","message":"invalid lat/lon","code":-1,"lang":lang_code,"nonce":now},
                        no_update, no_update, _inc_started(counters))
            deadline = now + 3.0
            return ({"type":"locate","lat":float(lat),"lon":float(lon),"lang":lang_code,"nonce":now},
                    {"rescan_pending": True, "rescan_until": deadline},
                    no_update,
                    _inc_started(counters))

        # 3) 定位失敗
        if trig == "loc_listener" and event and event.get("type")=="locationerror":
            msg = (event.get("message") or "").lower()
            code = event.get("code")
            return ({"type":"geo_error","message":msg,"code":code,"lang":lang_code,"nonce":now},
                    no_update, no_update, _inc_started(counters))

        # 4) moveend → 真實 bounds 精準重掃（限 3 秒內）
        if trig == "loc_listener" and event and event.get("type")=="moveend":
            ui = ui or {}
            deadline = float(ui.get("rescan_until") or 0.0)
            pending  = bool(ui.get("rescan_pending"))
            if pending and time.time() <= deadline:
                lat = lon = None
                if viewport: lat, lon = _vp_center(viewport)
                if (lat is None or lon is None) and map_center is not None:
                    lat, lon = _any_center(map_center)
                if lat is None or lon is None:
                    lat, lon = DEFAULT_TW_CENTER
                return ({"type":"area","lat":lat,"lon":lon,"lang":lang_code,"nonce":time.time()},
                        {"rescan_pending": False, "rescan_until": 0.0},
                        no_update,
                        _inc_started(counters))

        # 5) 在此區域搜尋（viewport.center → map.center → 台灣中心）
        if trig == "search-area-button":
            lat = lon = None
            if center_data and "lat" in center_data and "lon" in center_data:
                try: lat, lon = float(center_data["lat"]), float(center_data["lon"])
                except Exception: lat = lon = None
            if (lat is None or lon is None) and viewport:
                lat, lon = _vp_center(viewport)
            if (lat is None or lon is None) and map_center is not None:
                lat, lon = _any_center(map_center)
            if lat is None or lon is None:
                lat, lon = DEFAULT_TW_CENTER
            return ({"type":"area","lat":lat,"lon":lon,"lang":lang_code,"nonce":now},
                    no_update, no_update, _inc_started(counters))

        return no_update, no_update, no_update, no_update

    except Exception as e:
        logging.exception(f"dispatcher error: {e}")
        return ({"type":"geo_error","message":str(e),"code":-999,"lang":lang_code,"nonce":now},
                no_update, no_update, _inc_started(counters))

# 單一渲染器
@app.callback(
    Output('weather-layer','children'),
    Output('status-message','children'),
    Output('next-hour-alert','children'),
    Output('map','viewport'),
    Output('error-toast','children'),
    Output('error-toast','header'),
    Output('error-toast','is_open'),
    Output('location-store','data'),
    Output('ui-store','data', allow_duplicate=True),
    Output('task-counters','data', allow_duplicate=True),
    Input('action-store','data'),
    State('basemap-mode','value'),
    State('map','bounds'),
    State('map','zoom'),
    State('task-counters','data'),
    prevent_initial_call=True
)
def main_renderer(action, basemap_mode, bounds, zoom, counters):
    viewport_out = no_update
    toast_msg = toast_hd = no_update
    toast_open = False

    if not action:
        return no_update, no_update, no_update, viewport_out, no_update, no_update, False, no_update, no_update, no_update

    atype = action.get("type")
    lang_code = action.get("lang","zh")
    t = I18N.get(lang_code, I18N["zh"])

    if atype == "geo_error":
        code = action.get("code")
        msg_raw = (action.get("message") or "").lower()
        if code == 1 or "denied" in msg_raw:
            msg = t.get("geo_denied")
        elif code == 2 or "unavailable" in msg_raw:
            msg = t.get("geo_unavailable")
        elif code == 3 or "timeout" in msg_raw:
            msg = t.get("geo_timeout")
        else:
            msg = t.get("geo_error")
        return (no_update, no_update, no_update, viewport_out,
                msg, "Error", True,
                no_update, no_update, _inc_finished(counters))

    address = ""; lat = lon = None
    try:
        if atype == "search":
            q = (action.get("q") or "").strip()
            data = smart_geocode(q, lang_code)
            if not data:
                return (no_update, no_update, no_update, viewport_out,
                        t.get("err_not_found","Not found"), "Error", True,
                        {"address":"","lat":None,"lon":None,"nonce":time.time()},
                        no_update, _inc_finished(counters))
            address = data["address"]; lat = float(data["lat"]); lon = float(data["lon"])
            viewport_out = _safe_viewport(lat, lon, SEARCHED_ZOOM)

        elif atype == "locate":
            lat = float(action.get("lat")); lon = float(action.get("lon"))
            address = reverse_geocode(lat, lon, lang_code, prefer_area=False) or t.get("map_center","Map center")
            viewport_out = _safe_viewport(lat, lon, SEARCHED_ZOOM)

        elif atype == "area":
            lat = float(action.get("lat")); lon = float(action.get("lon"))
            address = (reverse_geocode(lat, lon, lang_code, prefer_area=True)
                       or reverse_geocode(lat, lon, lang_code, prefer_area=False)
                       or t.get("map_center","Map center"))
        else:
            return (no_update, no_update, no_update, viewport_out, no_update, no_update, False, no_update, no_update, _inc_finished(counters))
    except Exception as e:
        logging.exception(f"Action parse error: {e}")
        return (no_update, no_update, no_update, viewport_out,
                t.get("geo_error","Location failed."), "Error", True,
                no_update, no_update, _inc_finished(counters))

    # 面積掃描
    if atype in ("search", "locate"):
        scan_bounds = _bounds_from_center_zoom(lat, lon, SEARCHED_ZOOM)
        wx_points, step_used = get_weather_data_for_bounds(scan_bounds, SEARCHED_ZOOM)
    else:
        if bounds and zoom is not None:
            wx_points, step_used = get_weather_data_for_bounds(bounds, zoom)
        else:
            scan_bounds = _bounds_from_center_zoom(lat, lon, SEARCHED_ZOOM)
            wx_points, step_used = get_weather_data_for_bounds(scan_bounds, SEARCHED_ZOOM)

    if len(wx_points) > 300:
        stride = max(2, len(wx_points)//250)
        wx_points = wx_points[::stride]

    rain_layer = build_rain_circles(wx_points, step_used, basemap_mode) or []

    if atype in ("search","locate"):
        rain_layer.append(
            dl.Marker(position=[lat, lon],
                      children=[dl.Tooltip(address or "Here"),
                                dl.Popup(html.Div(address or "Here"))])
        )

    status = html.Div([
        dbc.Alert(address, color="info") if address else "",
        html.Small(f"({lat:.5f}, {lon:.5f})", className="text-muted ms-1"),
        html.Br(),
        html.Small(f"{t.get('data_updated','Updated at')} {datetime.now().strftime('%H:%M')}",
                   className="text-muted")
    ])

    mm_now, code_om, mm_next = _get_hourly_with_quantized_cache(lat, lon)
    try:
        alert = (dbc.Alert(f"☔ {t.get('maybe_rain','Possible rain')}：{float(mm_next):.1f} mm",
                           color="warning", className="mb-0")
                 if float(mm_next) >= ALERT_MM_MIN_MM else "")
    except Exception:
        alert = ""

    loc_data = {"address": address, "lat": lat, "lon": lon, "nonce": time.time()}

    return (rain_layer, status, alert, viewport_out,
            no_update, no_update, False,
            loc_data, no_update, _inc_finished(counters))

# --------------------------- 入口 ---------------------------
server = app.server  # ✅ 給 gunicorn 用

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=True, port=port, use_reloader=False)
