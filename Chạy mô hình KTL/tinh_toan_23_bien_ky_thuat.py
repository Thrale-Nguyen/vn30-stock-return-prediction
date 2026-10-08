# -*- coding: utf-8 -*-
"""
XÂY DỰNG 23 BIẾN ĐẶC TRƯNG KỸ THUẬT (FEATURE ENGINEERING)
Dữ liệu đầu vào: clean_data.csv
Dữ liệu đầu ra: feature_data.csv (Bảng dữ liệu mở rộng bổ sung 23 cột đặc trưng kỹ thuật)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Cấu hình UTF-8 cho console Windows
sys.stdout.reconfigure(encoding='utf-8')

# Danh sách 23 biến đặc trưng kỹ thuật chuẩn hóa theo Bảng 16 của SOP
FEATURE_COLS_23 = [
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
    # Nhóm 5: Khoảng cách đường MA (Moving Average Gaps - 3 biến)
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


def safe_div(a, b):
    """
    Phép chia an toàn tránh lỗi ZeroDivisionError hoặc chia cho NaN/Inf.
    Trả về NaN nếu mẫu số bằng 0 hoặc NaN.
    """
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isnan(b_arr) | (b_arr == 0), np.nan, a_arr / b_arr)


def compute_23_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán 23 biến đặc trưng kỹ thuật độc lập cho từng mã cổ phiếu (ticker).
    Tuân thủ nghiêm ngặt nguyên tắc chuỗi thời gian, không rò rỉ dữ liệu tương lai (No Look-ahead Bias).
    """
    df_work = df.copy()

    # Chuẩn hóa trường ngày giao dịch để sắp xếp đúng thứ tự thời gian tuyến tính
    if "parsed_date" not in df_work.columns:
        df_work["parsed_date"] = pd.to_datetime(df_work["time"], format="%d/%m/%Y", errors="coerce")

    parts = []
    # Lặp qua từng mã cổ phiếu để bảo đảm tính toán độc lập
    for ticker, group in df_work.groupby("ticker", sort=False):
        d = group.sort_values("parsed_date").copy().reset_index(drop=True)
        close = d["close"]
        open_p = d["open"]
        high = d["high"]
        low = d["low"]
        volume = d["volume"]

        # ==========================================================
        # Nhóm 1: Biến Lợi suất Quá khứ (Past Returns - 5 biến)
        # ==========================================================
        # 1. log_return: ln(Close_t / Close_{t-1})
        d["log_return"] = np.log(safe_div(close, close.shift(1)))
        # 2. return_1: (Close_t - Close_{t-1}) / Close_{t-1}
        d["return_1"] = close.pct_change(1)
        # 3. return_3: (Close_t - Close_{t-3}) / Close_{t-3}
        d["return_3"] = close.pct_change(3)
        # 4. return_5: (Close_t - Close_{t-5}) / Close_{t-5}
        d["return_5"] = close.pct_change(5)
        # 5. return_10: (Close_t - Close_{t-10}) / Close_{t-10}
        d["return_10"] = close.pct_change(10)

        # ==========================================================
        # Nhóm 2: Biến Động lượng Giá (Price Momentum - 1 biến)
        # ==========================================================
        # 6. momentum_20: ln(Close_t / Close_{t-20})
        d["momentum_20"] = np.log(safe_div(close, close.shift(20)))

        # ==========================================================
        # Nhóm 3: Biến Dao động Giá Trong Phiên (Intraday Price Variations - 4 biến)
        # ==========================================================
        # 7. oc_return: (Close_t - Open_t) / Open_t
        d["oc_return"] = safe_div(close - open_p, open_p)
        # 8. hl_range: (High_t - Low_t) / Close_t
        d["hl_range"] = safe_div(high - low, close)
        # 9. close_to_high: Close_t / High_t - 1
        d["close_to_high"] = safe_div(close, high) - 1
        # 10. close_to_low: Close_t / Low_t - 1
        d["close_to_low"] = safe_div(close, low) - 1

        # ==========================================================
        # Nhóm 4: Biến Thanh khoản Thị trường (Liquidity Dynamics - 3 biến)
        # ==========================================================
        # 11. volume_change: (Volume_t - Volume_{t-1}) / Volume_{t-1}
        d["volume_change"] = volume.pct_change(1)
        # 12. log_volume: ln(1 + Volume_t)
        d["log_volume"] = np.log1p(volume)
        # Biến phụ trợ thanh khoản 20 phiên: SMA_20(Volume)
        d["avg_volume_20"] = volume.rolling(20, min_periods=20).mean()
        # 13. volume_ratio_20: Volume_t / SMA_20(Volume)
        d["volume_ratio_20"] = safe_div(volume, d["avg_volume_20"])

        # ==========================================================
        # Nhóm 5: Biến Khoảng cách Trung bình Động (Moving Average Gaps - 3 biến)
        # Nhóm 6: Biến Độ Biến động Lịch sử (Historical Volatility - 3 biến)
        # ==========================================================
        # 14. ma5_gap, 15. ma10_gap, 16. ma20_gap: Close_t / SMA_k(Close) - 1
        # 17. volatility5, 18. volatility10, 19. volatility20: Độ lệch chuẩn mẫu log_return k phiên
        for w in [5, 10, 20]:
            ma = close.rolling(w, min_periods=w).mean()
            d[f"ma{w}_gap"] = safe_div(close, ma) - 1
            d[f"volatility{w}"] = d["log_return"].rolling(w, min_periods=w).std()

        # ==========================================================
        # Nhóm 7: Chỉ báo Kỹ thuật Nâng cao (Technical Indicators - 4 biến)
        # ==========================================================
        # 20. rsi14: Relative Strength Index 14 phiên (Wilder)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = safe_div(gain, loss)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # Xử lý trường hợp biên: 14 phiên toàn tăng (loss=0, gain>0) -> RSI=100
        rsi = np.where((loss == 0) & (gain > 0), 100.0, rsi)
        rsi = np.where((loss == 0) & (gain == 0), 50.0, rsi)
        d["rsi14"] = rsi

        # 21. macd: EMA_12(Close) - EMA_26(Close)
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        d["macd"] = ema12 - ema26
        # 22. macd_signal: EMA_9(MACD)
        d["macd_signal"] = d["macd"].ewm(span=9, adjust=False, min_periods=9).mean()

        # 23. bb_position: (Close - Lower_BB) / (Upper_BB - Lower_BB)
        ma20 = close.rolling(20, min_periods=20).mean()
        sd20 = close.rolling(20, min_periods=20).std()
        lower_bb = ma20 - 2.0 * sd20
        upper_bb = ma20 + 2.0 * sd20
        d["bb_position"] = safe_div(close - lower_bb, upper_bb - lower_bb)

        parts.append(d)

    result_df = pd.concat(parts, ignore_index=True)
    if "parsed_date" in result_df.columns:
        result_df = result_df.drop(columns=["parsed_date"])

    # Thay thế Inf / -Inf bằng NaN theo tiêu chuẩn tiền xử lý
    result_df = result_df.replace([np.inf, -np.inf], np.nan)

    # Đảm bảo cấu trúc các cột: 7 cột gốc + 23 biến kỹ thuật + avg_volume_20 (phục vụ bộ lọc Bước 12)
    base_cols = ["time", "open", "high", "low", "close", "volume", "ticker"]
    final_cols = base_cols + FEATURE_COLS_23 + ["avg_volume_20"]
    result_df = result_df[final_cols]

    return result_df


def main():
    workspace_dir = Path(__file__).resolve().parent
    input_file = workspace_dir / "clean_data.csv"
    output_file = workspace_dir / "feature_data.csv"
    output_alias = workspace_dir / "clean_data_features.csv"

    print("=" * 80)
    print("QUY TRÌNH THỰC NGHIỆM ĐỊNH LƯỢNG - BƯỚC 4: FEATURE ENGINEERING")
    print("=" * 80)
    print(f"- File dữ liệu đầu vào: {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file {input_file}!")

    df_raw = pd.read_csv(input_file)
    print(f"- Số quan sát đầu vào: {len(df_raw):,} dòng x {len(df_raw.columns)} cột")
    print(f"- Danh sách mã ({df_raw['ticker'].nunique()} mã): {sorted(df_raw['ticker'].unique())}")

    print("\nĐang tính toán 23 biến đặc trưng kỹ thuật cho từng mã cổ phiếu...")
    featured_df = compute_23_features(df_raw)

    print(f"\n=> KẾT QUẢ TÍNH TOÁN BƯỚC 4:")
    print(f"- Kích thước bảng dữ liệu mở rộng: {featured_df.shape[0]:,} dòng x {featured_df.shape[1]} cột")
    print(f"- 7 biến gốc ban đầu: {['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']}")
    print(f"- 23 biến kỹ thuật theo đúng SOP:")
    for idx, col in enumerate(FEATURE_COLS_23, start=1):
        print(f"  {idx:2d}. {col}")
    print(f"- 1 biến phụ trợ thanh khoản 20 phiên: avg_volume_20 (cho bộ lọc Bước 12)")

    # Kiểm tra tính toàn vẹn và tỷ lệ khuyết thiếu do cửa sổ trễ
    print("\n" + "=" * 80)
    print("THỐNG KÊ MÔ TẢ VÀ TỶ LỆ DỮ LIỆU KHUYẾT THIẾU CỦA 23 BIẾN ĐẶC TRƯNG")
    print("=" * 80)
    stats_df = pd.DataFrame({
        "STT": range(1, 24),
        "Mã biến": FEATURE_COLS_23,
        "Số mẫu hợp lệ": [featured_df[c].count() for c in FEATURE_COLS_23],
        "Số mẫu thiếu (NaN)": [featured_df[c].isna().sum() for c in FEATURE_COLS_23],
        "Tỷ lệ thiếu (%)": [round(featured_df[c].isna().mean() * 100, 2) for c in FEATURE_COLS_23],
        "Trung bình (Mean)": [round(featured_df[c].mean(), 4) for c in FEATURE_COLS_23],
        "Độ lệch chuẩn (Std)": [round(featured_df[c].std(), 4) for c in FEATURE_COLS_23],
        "Giá trị nhỏ nhất (Min)": [round(featured_df[c].min(), 4) for c in FEATURE_COLS_23],
        "Trung vị (50%)": [round(featured_df[c].median(), 4) for c in FEATURE_COLS_23],
        "Giá trị lớn nhất (Max)": [round(featured_df[c].max(), 4) for c in FEATURE_COLS_23],
    })
    print(stats_df.to_string(index=False))

    # Lưu file kết quả
    print(f"\nĐang lưu kết quả ra file: {output_file.name} ...")
    featured_df.to_csv(output_file, index=False, encoding="utf-8")
    featured_df.to_csv(output_alias, index=False, encoding="utf-8")
    print(f"✓ Đã lưu thành công: {output_file}")
    print(f"✓ Đã tạo bản sao đối chiếu: {output_alias}")
    print(f"✓ Kích thước file: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    print("\nHoàn tất BƯỚC 4 theo đúng quy chuẩn SOP!")


if __name__ == "__main__":
    main()
