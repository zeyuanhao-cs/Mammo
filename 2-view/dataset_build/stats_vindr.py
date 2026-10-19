# VinDr-Mammo 数据集统计：BI-RADS 五类、密度四类、各 finding 阳性数及全零数
import pandas as pd

LOCAL = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"
df = pd.read_csv(LOCAL)
print(f"Total annotation rows: {len(df)}")
print(f"Unique images: {df['image_id'].nunique()}")
print(f"Unique patients: {df['patient_id'].nunique()}")

print("\n=== breast_birads (image-level) ===")
print(df['breast_birads'].value_counts(dropna=False).sort_index())

print("\n=== breast_density ===")
print(df['breast_density'].value_counts(dropna=False).sort_index())

print("\n=== density (A-D) ===")
print(df['density'].value_counts(dropna=False).sort_index())

print("\n=== finding_birads (per-finding) ===")
print(df['finding_birads'].value_counts(dropna=False).sort_index())

finding_cols = ['Architectural_Distortion','Asymmetry','Focal_Asymmetry','Global_Asymmetry',
                'Mass','Nipple_Retraction','No_Finding','Skin_Retraction','Skin_Thickening',
                'Suspicious_Calcification','Suspicious_Lymph_Node']
print("\n=== Each finding positive count (rows where col==1) ===")
for c in finding_cols:
    print(f"{c:28s}: {(df[c]==1).sum()}")

positive_cols = [c for c in finding_cols if c != 'No_Finding']
all_zero = (df[positive_cols].sum(axis=1) == 0).sum()
print(f"\nAll-zero rows (no positive finding): {all_zero}")
print(f"No_Finding=1 rows: {(df['No_Finding']==1).sum()}")

# 图像级别去重统计（每张图取第一行的 breast_birads/density）
img = df.drop_duplicates(subset='image_id')
print("\n=== Image-level (dedup) breast_birads ===")
print(img['breast_birads'].value_counts(dropna=False).sort_index())
print("\n=== Image-level (dedup) breast_density ===")
print(img['breast_density'].value_counts(dropna=False).sort_index())

# 图像级别 finding 阳性数（该图任一行该列为1即算阳性）
img_finding = df.groupby('image_id')[finding_cols].max()
print("\n=== Image-level finding positive counts ===")
for c in finding_cols:
    print(f"{c:28s}: {int(img_finding[c].sum())}")
img_pos = img_finding.drop(columns=['No_Finding']).sum(axis=1)
print(f"\nImage-level all-zero (no positive finding): {(img_pos==0).sum()}")
