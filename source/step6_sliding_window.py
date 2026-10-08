# -*- coding: utf-8 -*-
"""
BƯỚC 6: THIẾT KẾ CẤU TRÚC MẪU CHUỖI THỜI GIAN THEO CỬA SỔ TRƯỢT (SLIDING WINDOW LOOKBACK)
Dự án: Nghiên cứu Dự báo xu hướng biến động lợi suất cổ phiếu VN30 bằng các mô hình học máy
Tài liệu quy chuẩn: research_instruction.docx (SOP BƯỚC 6) & Thuyết minh nghiên cứu Mục 3.5

Mục tiêu:
Chuyển hóa dữ liệu chuỗi thời gian dạng bảng phẳng thành các khối quan sát động học (Lookback Window),
bảo toàn thông tin lịch sử diễn biến giá và các mẫu hình kỹ thuật trong quá khứ gần.

Quy cách thiết lập cửa sổ trượt (Sliding Window):
• Độ dài cửa sổ quan sát (Lookback Length): W = 20 phiên giao dịch (tương đương 1 tháng làm việc).
• Số lượng đặc trưng mỗi phiên: M = 23 biến kỹ thuật chuẩn hóa từ Bước 4.
• Kích thước tensor quan sát ban đầu: Mỗi mẫu là ma trận 2D [20 x 23].
• Kỹ thuật làm phẳng đặc trưng (Flattening): Đối với các mô hình học máy dạng Tabular,
  ma trận [20 x 23] được trải phẳng thành một vector đặc trưng duy nhất có số chiều D = 20 x 23 = 460 chiều.
  Vector X_t được ghép cặp chính xác với nhãn Direction_{t+h} hình thành mẫu học (X_t, y_{t+h}).

Dữ liệu đầu vào: source/labeled_feature_data.csv
Dữ liệu đầu ra:
  - source/windowed_dataset_460.parquet (Ma trận đặc trưng 460 chiều + metadata + nhãn T+1, T+3, T+5)
  - source/windowed_data_summary.json (Báo cáo tóm tắt cấu trúc ma trận và phân phối nhãn)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Thiết lập mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================================================================
# HẰNG SỐ CẤU HÌNH THEO QUY CHUẨN SOP (BẢNG 16, BƯỚC 6)
# ==============================================================================

# Độ dài cửa sổ quan sát lịch sử (20 phiên giao dịch ~ 1 tháng làm việc)
LOOKBACK_WINDOW: int = 20

# Danh sách 23 biến đặc trưng kỹ thuật chuẩn hóa từ Bước 4
FEATURE_COLS_23: List[str] = [
    # Nhóm 1: Lợi suất quá khứ (5 biến)
    "log_return",
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    # Nhóm 2: Động lượng giá (1 biến)
    "momentum_20",
    # Nhóm 3: Dao động phiên (4 biến)
    "oc_return",
    "hl_range",
    "close_to_high",
    "close_to_low",
    # Nhóm 4: Thanh khoản thị trường (3 biến)
    "volume_change",
    "log_volume",
    "volume_ratio_20",
    # Nhóm 5: Khoảng cách đường MA (3 biến)
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    # Nhóm 6: Độ biến động lịch sử (3 biến)
    "volatility5",
    "volatility10",
    "volatility20",
    # Nhóm 7: Chỉ báo kỹ thuật nâng cao (4 biến)
    "rsi14",
    "macd",
    "macd_signal",
    "bb_position",
]

# Các chân trời dự báo tương lai theo SOP (Bước 5)
HORIZONS: List[int] = [1, 3, 5]

# Cấu hình kiểm soát tính liên tục lịch giao dịch (Trading Calendar Continuity Control)
# 20 phiên giao dịch thông thường kéo dài 26-28 ngày dương lịch.
# Ngưỡng tối đa 40 ngày (hoặc 35 ngày) cho phép bao trọn các kỳ nghỉ lễ dài như Tết Nguyên Đán.
DEFAULT_MAX_CALENDAR_SPAN_DAYS: int = 40
STRICT_MAX_CALENDAR_SPAN_DAYS: int = 35

# Các trường định danh và nhãn mục tiêu cần bảo lưu
METADATA_COLS: List[str] = [
    "time",
    "parsed_date",
    "ticker",
    "close",
    "window_start_date",
    "calendar_span_days",
    "is_continuous_window",
]

TARGET_LABEL_COLS: List[str] = [
    "direction_t1",
    "future_return_t1",
    "target_date_t1",
    "direction_t3",
    "future_return_t3",
    "target_date_t3",
    "direction_t5",
    "future_return_t5",
    "target_date_t5",
]


def generate_feature_column_names(
    feature_cols: List[str] = FEATURE_COLS_23,
    lookback: int = LOOKBACK_WINDOW,
    style: str = "t_minus",
) -> List[str]:
    """
    Sinh danh sách tên chuẩn cho D = W x M đặc trưng sau khi trải phẳng (Flattening).
    
    Quy cách sắp xếp thời gian:
    Từ quá khứ xa nhất (t - 19) đến hiện tại (t):
    - Phiên t - 19: {feat}_t_minus_19 (hoặc {feat}_lag19)
    - ...
    - Phiên t - 1:  {feat}_t_minus_1  (hoặc {feat}_lag1)
    - Phiên t:      {feat}_t          (hoặc {feat}_lag0)
    
    Parameters:
        feature_cols: Danh sách M đặc trưng kỹ thuật mỗi phiên (mặc định 23 biến).
        lookback: Độ dài cửa sổ quan sát W (mặc định 20 phiên).
        style: Phong cách đặt tên ('t_minus' hoặc 'lag').
        
    Returns:
        List[str]: Danh sách gồm D = W x M tên cột đặc trưng.
    """
    names: List[str] = []
    for tau in range(lookback - 1, -1, -1):
        if style == "t_minus":
            suffix = f"_t_minus_{tau}" if tau > 0 else "_t"
        elif style == "lag":
            suffix = f"_lag{tau}"
        else:
            suffix = f"_{tau}"
            
        for f in feature_cols:
            names.append(f"{f}{suffix}")
            
    return names


def create_sliding_windows(
    df: pd.DataFrame,
    lookback: int = LOOKBACK_WINDOW,
    feature_cols: List[str] = FEATURE_COLS_23,
    style: str = "t_minus",
    max_calendar_span_days: int = DEFAULT_MAX_CALENDAR_SPAN_DAYS,
    filter_irregular_windows: bool = False,
) -> pd.DataFrame:
    """
    Chuyển đổi dữ liệu chuỗi thời gian thành ma trận đặc trưng cửa sổ trượt (Sliding Window).
    
    Quy trình toán học & xử lý dữ liệu:
    1. Chuẩn hóa ngày giao dịch và sắp xếp chuỗi thời gian tăng dần độc lập cho từng mã cổ phiếu.
    2. Với mỗi mã cổ phiếu i có chuỗi chiều dài L_i:
       - Tạo các khối tensor [20 x 23] từ phiên t-19 đến phiên t thông qua kỹ thuật
         vectorization sliding_window_view hiệu năng cao của NumPy.
       - Trải phẳng ma trận [20 x 23] thành vector D = 460 chiều:
         X_t = [f_1(t-19), ..., f_23(t-19), ..., f_1(t), ..., f_23(t)].
       - Trích xuất thông tin metadata và nhãn mục tiêu tại thời điểm t (kết thúc cửa sổ).
    3. Kiểm soát tính liên tục lịch giao dịch (Trading Calendar Continuity Control):
       - Đo lường khoảng cách ngày dương lịch thực tế giữa đầu và cuối cửa sổ:
         calendar_span_days = (date_t - date_{t-19}).days
       - Với 20 phiên thông thường, khoảng cách chuẩn là 26-28 ngày, tối đa cho phép 35-40 ngày
         (đã bao gồm các kỳ nghỉ lễ dài ngày như Tết Nguyên Đán).
       - Gắn cờ is_continuous_window = (calendar_span_days <= max_calendar_span_days)
         và lưu ngày bắt đầu cửa sổ window_start_date.
    4. Hợp nhất tất cả các mã cổ phiếu và sắp xếp toàn diện theo thứ tự thời gian tuyến tính
       (parsed_date, ticker) phục vụ việc phân tách mẫu Chronological Split tại Bước 7.
       
    Parameters:
        df: DataFrame chứa dữ liệu 23 biến kỹ thuật và nhãn mục tiêu (labeled_feature_data.csv).
        lookback: Độ dài cửa sổ quan sát (W = 20).
        feature_cols: Danh sách 23 biến đặc trưng.
        style: Phong cách đặt tên cột ('t_minus' hoặc 'lag').
        max_calendar_span_days: Ngưỡng ngày dương lịch tối đa cho phép giữa t-19 và t (mặc định 40 ngày).
        filter_irregular_windows: Nếu True, chỉ giữ lại các cửa sổ liên tục đạt chuẩn.
        
    Returns:
        pd.DataFrame: Bảng dữ liệu hoàn chỉnh với 460 cột đặc trưng X_t và các cột nhãn, metadata.
    """
    df_work = df.copy()

    # Chuẩn hóa trường ngày giao dịch để đảm bảo sắp xếp chuỗi thời gian chính xác
    if "parsed_date" not in df_work.columns:
        df_work["parsed_date"] = pd.to_datetime(
            df_work["time"], format="%d/%m/%Y", errors="coerce"
        )
    else:
        df_work["parsed_date"] = pd.to_datetime(df_work["parsed_date"])

    # Danh sách 460 tên cột đặc trưng trải phẳng
    feature_names = generate_feature_column_names(
        feature_cols=feature_cols, lookback=lookback, style=style
    )

    parts: List[pd.DataFrame] = []
    total_tickers = df_work["ticker"].nunique()
    print(f"[*] Bắt đầu khởi tạo Sliding Window (W={lookback}, M={len(feature_cols)}, D={len(feature_names)})...")
    print(f"[*] Tổng số mã cổ phiếu xử lý: {total_tickers} mã.")
    print(f"[*] Cơ chế kiểm soát lịch giao dịch: Ngưỡng tối đa {max_calendar_span_days} ngày dương lịch.")

    start_time = time.time()

    # Lặp qua từng mã cổ phiếu để xử lý cửa sổ trượt độc lập, chống rò rỉ xuyên mã (Cross-ticker Leakage)
    for ticker, group in df_work.groupby("ticker", sort=False):
        g = group.sort_values("parsed_date").reset_index(drop=True)
        L = len(g)

        if L < lookback:
            print(f"[!] Cảnh báo: Mã {ticker} chỉ có {L} phiên < {lookback} phiên. Bỏ qua.")
            continue

        # Trích xuất ma trận giá trị 23 biến đặc trưng dưới dạng float32 để tối ưu hóa bộ nhớ RAM
        vals = g[feature_cols].to_numpy(dtype=np.float32)

        # Sử dụng sliding_window_view để trượt cửa sổ không tốn chi phí sao chép bộ nhớ
        # Output shape: (num_windows, 23, lookback)
        window_views = np.lib.stride_tricks.sliding_window_view(
            vals, window_shape=lookback, axis=0
        )

        # Chuyển đổi trục sang (num_windows, lookback, 23) và trải phẳng thành (num_windows, 460)
        flattened_windows = np.transpose(window_views, (0, 2, 1)).reshape(
            len(window_views), -1
        )

        # Trích xuất chuỗi thời gian để tính toán tính liên tục lịch giao dịch (Trading Calendar Continuity Control)
        dates_arr = g["parsed_date"].values  # np.datetime64[ns]
        date_views = np.lib.stride_tricks.sliding_window_view(
            dates_arr, window_shape=lookback
        )
        # date_views: (num_windows, lookback) - Cột 0 là t - 19, Cột -1 là t
        span_days = (date_views[:, -1] - date_views[:, 0]) / np.timedelta64(1, "D")
        is_continuous = span_days <= max_calendar_span_days
        window_start_dates = pd.to_datetime(date_views[:, 0]).strftime("%d/%m/%Y")

        # Trích xuất thông tin metadata và nhãn mục tiêu tại phiên kết thúc cửa sổ (phiên t)
        meta_slice = g.iloc[lookback - 1 :].copy().reset_index(drop=True)

        # Khởi tạo DataFrame cho mã hiện tại
        ticker_df = pd.DataFrame(flattened_windows, columns=feature_names)

        # Đính kèm metadata
        ticker_df["time"] = meta_slice["time"].values
        ticker_df["parsed_date"] = meta_slice["parsed_date"].values
        ticker_df["ticker"] = meta_slice["ticker"].values
        ticker_df["close"] = meta_slice["close"].values
        ticker_df["window_start_date"] = window_start_dates
        ticker_df["calendar_span_days"] = span_days.astype(np.int32)
        ticker_df["is_continuous_window"] = is_continuous

        # Đính kèm các biến mục tiêu và lợi suất tương lai
        for col in TARGET_LABEL_COLS:
            if col in meta_slice.columns:
                ticker_df[col] = meta_slice[col].values

        if filter_irregular_windows:
            ticker_df = ticker_df[ticker_df["is_continuous_window"]].reset_index(drop=True)

        parts.append(ticker_df)

    # Ghép nối tất cả các mã cổ phiếu
    final_df = pd.concat(parts, ignore_index=True)

    # Sắp xếp toàn diện theo trục thời gian tuyến tính (parsed_date), sau đó theo ticker
    final_df = final_df.sort_values(["parsed_date", "ticker"]).reset_index(drop=True)

    elapsed = time.time() - start_time
    print(f"[✓] Đã tạo thành công {len(final_df):,} mẫu quan sát trong {elapsed:.2f} giây.")
    print(f"[✓] Kích thước ma trận tổng thể: {final_df.shape[0]} hàng x {final_df.shape[1]} cột.")

    return final_df


def extract_X_y_for_horizon(
    windowed_df: pd.DataFrame,
    horizon: int = 5,
    feature_cols: Optional[List[str]] = None,
    drop_na_target: bool = True,
    only_continuous: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Trích xuất ma trận đặc trưng X và vector nhãn y cho một chân trời dự báo cụ thể (T+1, T+3, T+5).
    
    Parameters:
        windowed_df: DataFrame thu được từ create_sliding_windows().
        horizon: Kỳ dự báo h (1, 3 hoặc 5).
        feature_cols: Danh sách 460 cột đặc trưng (nếu None sẽ tự động lọc các cột chứa '_t').
        drop_na_target: Nếu True, loại bỏ các dòng bị NaN ở biến mục tiêu do xử lý biên tương lai.
        only_continuous: Nếu True, chỉ giữ lại các cửa sổ thỏa mãn tính liên tục lịch dương (<= 40 ngày).
        
    Returns:
        Tuple[X, y, meta_df]:
          - X: Ma trận NumPy [N x 460], dtype float32.
          - y: Vector nhãn nhị phân NumPy [N], dtype int32.
          - meta_df: DataFrame chứa ticker, time, close, calendar_span_days, future_return tương ứng.
    """
    target_col = f"direction_t{horizon}"
    if target_col not in windowed_df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn '{target_col}' trong DataFrame.")

    df_sub = windowed_df.copy()
    if drop_na_target:
        df_sub = df_sub.dropna(subset=[target_col]).reset_index(drop=True)

    if only_continuous and "is_continuous_window" in df_sub.columns:
        df_sub = df_sub[df_sub["is_continuous_window"] == True].reset_index(drop=True)

    if feature_cols is None:
        # Lấy tất cả các cột không thuộc metadata và target
        excluded = set(METADATA_COLS + TARGET_LABEL_COLS)
        feature_cols = [c for c in df_sub.columns if c not in excluded]

    X = df_sub[feature_cols].to_numpy(dtype=np.float32)
    y = df_sub[target_col].to_numpy(dtype=np.int32)
    
    meta_cols_to_keep = [c for c in METADATA_COLS if c in df_sub.columns] + [f"future_return_t{horizon}"]
    meta_df = df_sub[meta_cols_to_keep].copy()

    return X, y, meta_df


def compute_dataset_summary(windowed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tính toán các chỉ số thống kê và phân phối nhãn cho báo cáo tóm tắt cấu trúc ma trận.
    Bao gồm kiểm toán tính liên tục lịch giao dịch (Trading Calendar Continuity).
    """
    excluded = set(METADATA_COLS + TARGET_LABEL_COLS)
    feature_cols = [c for c in windowed_df.columns if c not in excluded]

    summary = {
        "total_samples": int(len(windowed_df)),
        "num_features": int(len(feature_cols)),
        "lookback_window": int(LOOKBACK_WINDOW),
        "num_base_features": int(len(FEATURE_COLS_23)),
        "num_tickers": int(windowed_df["ticker"].nunique()),
        "ticker_list": sorted(windowed_df["ticker"].unique().tolist()),
        "start_date": str(windowed_df["parsed_date"].min().strftime("%Y-%m-%d")),
        "end_date": str(windowed_df["parsed_date"].max().strftime("%Y-%m-%d")),
        "samples_per_horizon": {},
    }

    # Thống kê tính liên tục lịch giao dịch (Trading Calendar Continuity)
    if "calendar_span_days" in windowed_df.columns:
        spans = windowed_df["calendar_span_days"]
        n_continuous = int((windowed_df["is_continuous_window"] == True).sum()) if "is_continuous_window" in windowed_df.columns else int(len(windowed_df))
        summary["calendar_continuity"] = {
            "max_allowed_span_days": int(DEFAULT_MAX_CALENDAR_SPAN_DAYS),
            "continuous_samples": n_continuous,
            "irregular_samples": int(len(windowed_df) - n_continuous),
            "continuity_ratio_pct": round(n_continuous / len(windowed_df) * 100, 2),
            "span_days_min": int(spans.min()),
            "span_days_mean": float(round(spans.mean(), 2)),
            "span_days_median": int(spans.median()),
            "span_days_max": int(spans.max()),
            "spans_above_35_days": int((spans > 35).sum()),
            "spans_above_40_days": int((spans > 40).sum()),
        }

    for h in HORIZONS:
        target_col = f"direction_t{h}"
        ret_col = f"future_return_t{h}"
        valid_series = windowed_df[target_col].dropna()
        n_valid = int(len(valid_series))
        n_pos = int((valid_series == 1).sum())
        n_neg = int((valid_series == 0).sum())
        pos_ratio = round(n_pos / n_valid * 100, 2) if n_valid > 0 else 0.0

        summary["samples_per_horizon"][f"T+{h}"] = {
            "valid_samples": n_valid,
            "nan_edge_samples": int(windowed_df[target_col].isna().sum()),
            "class_1_increase": n_pos,
            "class_0_non_increase": n_neg,
            "positive_ratio_pct": pos_ratio,
            "mean_future_log_return": float(round(windowed_df[ret_col].dropna().mean(), 6)),
        }

    return summary


def run_pipeline() -> None:
    """
    Hàm thực thi toàn bộ pipeline Bước 6:
    1. Đọc tệp đầu vào labeled_feature_data.csv.
    2. Chuyển đổi cửa sổ trượt Lookback W = 20 thành 460 chiều.
    3. Kiểm soát tính liên tục lịch giao dịch (Trading Calendar Continuity Control).
    4. Thực hiện kiểm chứng tính toàn vẹn dữ liệu (Sanity Checks).
    5. Xuất tệp windowed_dataset_460.parquet và windowed_data_summary.json.
    """
    project_root = Path(__file__).resolve().parent.parent
    source_dir = project_root / "source"
    input_file = source_dir / "labeled_feature_data.csv"
    output_parquet = source_dir / "windowed_dataset_460.parquet"
    output_summary = source_dir / "windowed_data_summary.json"

    print("=" * 80)
    print("BƯỚC 6: THIẾT KẾ CẤU TRÚC MẪU CHUỖI THỜI GIAN THEO CỬA SỔ TRƯỢT")
    print(f"[*] Tệp đầu vào: {input_file}")
    print(f"[*] Tệp đầu ra Parquet: {output_parquet}")
    print("=" * 80)

    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào tại: {input_file}")

    # Đọc dữ liệu đầu vào
    print(f"[*] Đang đọc dữ liệu từ {input_file.name}...")
    df_raw = pd.read_csv(input_file)
    print(f"[✓] Đã đọc {len(df_raw):,} dòng từ {df_raw['ticker'].nunique()} mã cổ phiếu.")

    # Thực thi biến đổi cửa sổ trượt có kiểm soát tính liên tục lịch giao dịch
    df_windowed = create_sliding_windows(
        df=df_raw,
        lookback=LOOKBACK_WINDOW,
        feature_cols=FEATURE_COLS_23,
        style="t_minus",
        max_calendar_span_days=DEFAULT_MAX_CALENDAR_SPAN_DAYS,
        filter_irregular_windows=False,
    )

    # Kiểm tra chất lượng và tính toàn vẹn dữ liệu (Sanity Checks)
    print("\n" + "-" * 40)
    print("[*] KIỂM TRA CHẤT LƯỢNG VÀ TOÀN VẸN DỮ LIỆU (SANITY CHECKS):")
    expected_dim = LOOKBACK_WINDOW * len(FEATURE_COLS_23)
    feature_cols_in_df = [
        c for c in df_windowed.columns if c not in set(METADATA_COLS + TARGET_LABEL_COLS)
    ]
    assert len(feature_cols_in_df) == expected_dim, (
        f"Lỗi: Số chiều đặc trưng {len(feature_cols_in_df)} != {expected_dim}"
    )
    print(f"  [✓] Số chiều vector đặc trưng D: {len(feature_cols_in_df)} (khớp hoàn hảo 20 x 23 = 460)")

    # Kiểm tra số lượng mẫu bị trừ do cửa sổ 20 phiên đầu: 25 mã x 19 phiên = 475 phiên
    expected_rows = len(df_raw) - (df_raw["ticker"].nunique() * (LOOKBACK_WINDOW - 1))
    assert len(df_windowed) == expected_rows, (
        f"Lỗi: Số dòng {len(df_windowed)} != kỳ vọng {expected_rows}"
    )
    print(f"  [✓] Tổng số mẫu hợp lệ sau khi trừ 19 phiên khởi tạo: {len(df_windowed):,} mẫu")

    # Kiểm tra tính liên tục lịch giao dịch (Trading Calendar Continuity)
    summary = compute_dataset_summary(df_windowed)
    if "calendar_continuity" in summary:
        cal = summary["calendar_continuity"]
        print(f"  [✓] Tính liên tục lịch giao dịch: {cal['continuous_samples']:,} / {len(df_windowed):,} mẫu đạt chuẩn (<= {cal['max_allowed_span_days']} ngày) ({cal['continuity_ratio_pct']}%)")
        print(f"      Khoảng cách lịch dương: Trung vị = {cal['span_days_median']} ngày | TB = {cal['span_days_mean']} ngày | Min = {cal['span_days_min']} ngày | Max = {cal['span_days_max']} ngày")
        if cal["irregular_samples"] > 0:
            print(f"      [!] Phát hiện {cal['irregular_samples']} mẫu vượt ngưỡng {cal['max_allowed_span_days']} ngày (Cửa sổ có kỳ nghỉ dài/tạm ngừng giao dịch được gắn cờ is_continuous_window=False)")

    # Kiểm tra phân phối nhãn cho từng chân trời
    for h in HORIZONS:
        info = summary["samples_per_horizon"][f"T+{h}"]
        print(f"  [✓] Kỳ T+{h}: {info['valid_samples']:,} mẫu hợp lệ | "
              f"Tăng (1): {info['class_1_increase']:,} ({info['positive_ratio_pct']}%) | "
              f"Không tăng (0): {info['class_0_non_increase']:,} | "
              f"Biên NaN: {info['nan_edge_samples']} mẫu")

    # Lưu trữ dữ liệu dạng Parquet nén Snappy siêu tốc
    print("\n" + "-" * 40)
    print(f"[*] Đang lưu tập dữ liệu dạng Parquet: {output_parquet.name}...")
    df_windowed.to_parquet(output_parquet, compression="snappy", index=False)
    parquet_size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"[✓] Đã xuất thành công: {output_parquet} ({parquet_size_mb:.2f} MB)")

    # Lưu tệp JSON tóm tắt cấu trúc ma trận
    print(f"[*] Đang lưu báo cáo tóm tắt: {output_summary.name}...")
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[✓] Đã lưu báo cáo tóm tắt: {output_summary}")

    print("=" * 80)
    print("HOÀN TẤT THÀNH CÔNG BƯỚC 6 (THIẾT KẾ CẤU TRÚC MẪU CHUỖI THỜI GIAN THEO CỬA SỔ TRƯỢT)")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
