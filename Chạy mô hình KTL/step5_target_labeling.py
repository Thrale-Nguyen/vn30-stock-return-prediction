# -*- coding: utf-8 -*-
"""
BƯỚC 5: ĐỊNH NGHĨA VÀ XÂY DỰNG BIẾN MỤC TIÊU PHÂN LOẠI (TARGET LABEL GENERATION)
Dự án: Nghiên cứu Dự báo xu hướng biến động lợi suất cổ phiếu VN30 bằng các mô hình học máy
Tài liệu quy chuẩn: research_instruction.docx (SOP BƯỚC 5 - Bảng 17, 18, 19)
Dữ liệu đầu vào: feature_data.csv (Bộ 23 biến kỹ thuật từ Bước 4)
Dữ liệu đầu ra: labeled_feature_data.csv (Bảng dữ liệu hoàn chỉnh gồm 23 đặc trưng + nhãn xu hướng T+1, T+3, T+5)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Cấu hình UTF-8 cho console Windows
sys.stdout.reconfigure(encoding='utf-8')

# Các chân trời dự báo theo quy chuẩn SOP
HORIZONS = [1, 3, 5]


def safe_div(a, b):
    """
    Phép chia an toàn tránh lỗi ZeroDivisionError hoặc chia cho NaN/Inf.
    """
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isnan(b_arr) | (b_arr == 0), np.nan, a_arr / b_arr)


def add_target_labels(df: pd.DataFrame, horizons: list = [1, 3, 5]) -> pd.DataFrame:
    """
    Xác định biến phụ thuộc nhị phân Direction_{t+h} cho từng chân trời h in {1, 3, 5}.
    
    Công thức toán học (Bảng 17 & 18):
    1. Lợi suất logarit tương lai:
       R(i, t+h) = ln( Close(i, t+h) / Close(i, t) )
    2. Quy tắc gán nhãn xu hướng nhị phân:
       Direction(i, t+h) = 1 nếu R(i, t+h) > 0  (TĂNG GIÁ)
       Direction(i, t+h) = 0 nếu R(i, t+h) <= 0 (KHÔNG TĂNG: Giảm hoặc Đi ngang)
       
    Quy tắc chống rò rỉ dữ liệu biên (Bảng 19):
    Tại h phiên cuối cùng của mỗi mã cổ phiếu, Close(i, t+h) chưa diễn ra trong thực tế,
    do đó Direction(i, t+h) và R(i, t+h) bắt buộc phải là NaN.
    """
    df_work = df.copy()

    # Chuẩn hóa trường ngày giao dịch để bảo đảm sắp xếp chuỗi thời gian chính xác
    if "parsed_date" not in df_work.columns:
        df_work["parsed_date"] = pd.to_datetime(df_work["time"], format="%d/%m/%Y", errors="coerce")

    parts = []
    # Lặp qua từng mã cổ phiếu để xử lý biên độc lập cho từng mã
    for ticker, group in df_work.groupby("ticker", sort=False):
        d = group.sort_values("parsed_date").copy().reset_index(drop=True)
        close = d["close"]
        time_col = d["time"]

        for h in horizons:
            # Dịch chuyển giá đóng cửa tương lai và ngày tương lai
            future_close = close.shift(-h)
            target_date = time_col.shift(-h)

            # Lợi suất logarit tương lai chân trời h phiên: R(i, t+h) = ln( Close_{t+h} / Close_t )
            future_return = np.log(safe_div(future_close, close))

            # Gán nhãn nhị phân: 1 nếu R > 0; 0 nếu R <= 0; NaN nếu chưa có dữ liệu tương lai
            direction = np.where(
                np.isnan(future_return),
                np.nan,
                np.where(future_return > 0, 1.0, 0.0)
            )

            # Lưu vào dataframe với tên chuẩn hóa
            d[f"target_date_t{h}"] = target_date
            d[f"future_return_t{h}"] = future_return
            d[f"direction_t{h}"] = direction

        parts.append(d)

    result_df = pd.concat(parts, ignore_index=True)
    if "parsed_date" in result_df.columns:
        result_df = result_df.drop(columns=["parsed_date"])

    return result_df


def main():
    workspace_dir = Path(__file__).resolve().parent
    input_file = workspace_dir / "feature_data.csv"
    output_file = workspace_dir / "labeled_feature_data.csv"
    output_alias = workspace_dir / "feature_data_with_target.csv"

    print("=" * 80)
    print("QUY TRÌNH THỰC NGHIỆM ĐỊNH LƯỢNG - BƯỚC 5: TẠO BIẾN MỤC TIÊU (TARGET LABELING)")
    print("=" * 80)
    print(f"- File dữ liệu đầu vào: {input_file.name}")

    if not input_file.exists():
        # Thử tìm file thay thế nếu có
        alt_input = workspace_dir / "clean_data_features.csv"
        if alt_input.exists():
            input_file = alt_input
        else:
            raise FileNotFoundError(f"Không tìm thấy file {input_file}!")

    df = pd.read_csv(input_file)
    print(f"- Số quan sát đầu vào: {len(df):,} dòng x {len(df.columns)} cột")
    print(f"- Số lượng mã cổ phiếu: {df['ticker'].nunique()} mã")
    print(f"- 3 chân trời dự báo: T+1 (h=1), T+3 (h=3), T+5 (h=5)")

    print("\nĐang tính toán lợi suất tương lai R_{t+h} và gán nhãn Direction_{t+h}...")
    labeled_df = add_target_labels(df, horizons=HORIZONS)

    print(f"\n=> KẾT QUẢ XÂY DỰNG BƯỚC 5:")
    print(f"- Kích thước bảng dữ liệu sau gán nhãn: {labeled_df.shape[0]:,} dòng x {labeled_df.shape[1]} cột")

    # Kiểm tra phân phối nhãn tại từng chân trời
    print("\n" + "=" * 80)
    print("THỐNG KÊ PHÂN PHỐI NHÃN MỤC TIÊU PHÂN LOẠI (DIRECTION LABELS)")
    print("=" * 80)

    summary_rows = []
    for h in HORIZONS:
        col_dir = f"direction_t{h}"
        col_ret = f"future_return_t{h}"
        
        valid_series = labeled_df[col_dir].dropna()
        n_valid = int(len(valid_series))
        n_nan = int(labeled_df[col_dir].isna().sum())
        n_up = int((valid_series == 1.0).sum())
        n_non_up = int((valid_series == 0.0).sum())
        pct_up = (n_up / n_valid) * 100.0 if n_valid > 0 else 0
        pct_non_up = (n_non_up / n_valid) * 100.0 if n_valid > 0 else 0

        ret_mean = labeled_df[col_ret].mean()
        ret_std = labeled_df[col_ret].std()
        ret_min = labeled_df[col_ret].min()
        ret_max = labeled_df[col_ret].max()

        summary_rows.append({
            "Kỳ dự báo": f"T+{h}",
            "Chân trời (h)": h,
            "Số mẫu hợp lệ": f"{n_valid:,}",
            "Số mẫu NaN (đuôi)": f"{n_nan:,}",
            "Lớp 1 (Tăng)": f"{n_up:,} ({pct_up:.2f}%)",
            "Lớp 0 (Không tăng)": f"{n_non_up:,} ({pct_non_up:.2f}%)",
            "Tỷ lệ Cân bằng Lớp": f"{n_up / n_non_up:.3f}",
            "Mean Return": f"{ret_mean:.4f}",
            "Std Return": f"{ret_std:.4f}",
            "Min Return": f"{ret_min:.4f}",
            "Max Return": f"{ret_max:.4f}",
        })

    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))

    # Lưu dữ liệu
    print(f"\nĐang lưu kết quả ra file: {output_file.name} ...")
    labeled_df.to_csv(output_file, index=False, encoding="utf-8")
    labeled_df.to_csv(output_alias, index=False, encoding="utf-8")

    print(f"✓ Đã lưu thành công: {output_file}")
    print(f"✓ Đã tạo bản sao đối chiếu: {output_alias}")
    print(f"✓ Dung lượng file: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    print("\nHoàn tất BƯỚC 5 theo đúng chuẩn SOP!")


if __name__ == "__main__":
    main()
