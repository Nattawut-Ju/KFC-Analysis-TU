"""
============================================================
สมมติฐานข้อที่ 2 — โค้ดฉบับสมบูรณ์ (เพิ่ม 3 ตัวแปรพฤติกรรม)
Pooled Logit Model with Clustered Standard Errors (by ID)
ใช้กลุ่ม B : +30, +50, +70 บาท (มีของแถม)

ตัวแปร Control เพิ่มเติม:
  - ความถี่ในการซื้อ KFC ต่อเดือน
  - ช่องทางหลักในการซื้อ KFC
  - ประเภทเมนูที่ซื้อปกติ
============================================================
วิธีใช้:
1. วางไฟล์ Excel ไว้ในโฟลเดอร์เดียวกับไฟล์ .py นี้
2. รันคำสั่ง: python hypothesis2_with_behavior.py
============================================================
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# ============================================================
# Step 1 : โหลดและเตรียมข้อมูล
# ============================================================
FILE_PATH = r"C:\Users\USER\Downloads\แบบสอบถาม_KFC_489.xlsx"
# ถ้าไฟล์อยู่ที่อื่น แก้ path ด้านบน เช่น:
# FILE_PATH = r"C:\Users\YourName\Downloads\แบบสอบถาม...xlsx"

df = pd.read_excel(FILE_PATH)
df.columns = df.columns.str.strip()
print(f"โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

# เปลี่ยนชื่อคอลัมน์ราคากลุ่ม B
rename_dict = {
    'หากราคาปรับขึ้นเป็น 229 บาท (+30 บาท) จะได้ไก่ทอดเพิ่ม 1 ชิ้น คุณจะยังซื้ออยู่ไหม?': 'add30',
    'หากราคาปรับขึ้นเป็น 249 บาท (+50 บาท) จะได้นักเก็ตส์เพิ่ม อีก 6 ชิ้น คุณจะยังซื้ออยู่ไหม?': 'add50',
    'หากราคาปรับขึ้นเป็น 269 บาท (+70 บาท) จะได้ไก่ทอดเพิ่ม 1 ชิ้น และ วิงแซ่บเพิ่ม 2 ชิ้น คุณจะยังซื้ออยู่ไหม': 'add70',
}
df.rename(columns=rename_dict, inplace=True)

for col in ['add30', 'add50', 'add70']:
    df[col] = df[col].astype(str).str.strip().map({'ซื้อ': 1, 'ไม่ซื้อ': 0})
df = df.dropna(subset=['add30', 'add50', 'add70'])
for col in ['add30', 'add50', 'add70']:
    df[col] = df[col].astype(int)
df['ID'] = range(1, len(df) + 1)
print(f"จำนวนผู้ตอบแบบสอบถาม: {len(df)} คน")

# ============================================================
# Step 2 : ยุบกลุ่ม 3 ตัวแปรพฤติกรรม (ก่อน melt)
# ============================================================
freq_col    = 'ความถี่ในการซื้อผลิตภัณฑ์จาก KFC ต่อเดือน'
channel_col = 'ช่องทางหลักในการซื้อ KFC'
menu_col    = 'ปกติแล้วท่านตัดสินใจซื้อเมนูใดของ KFC แบบใด'

# ยุบกลุ่มความถี่ (base = 1 ครั้ง)
df['Freq_Group'] = df[freq_col].replace({
    'ไม่เคยซื้อ':    'Rarely',
    '1 ครั้ง':       '1 time',
    '2-3 ครั้ง':     '2-3 times',
    '4 ครั้งขึ้นไป': '4+ times',
})

# ยุบกลุ่มช่องทาง (base = In-store)
df['Channel_Group'] = df[channel_col].apply(lambda x:
    'App/Online' if ('แอปพลิเคชัน' in str(x) and 'หน้าร้าน' not in str(x))
                    or 'จัดส่ง' in str(x)
    else 'In-store'
)

# ยุบกลุ่มเมนู (base = Bundle)
df['Menu_Group'] = df[menu_col].replace({
    'เมนูชุด (Bundling set)': 'Bundle',
    'เมนูเดี่ยว (A la carte)': 'A-la-carte',
    'ทั้งสองแบบ':             'Both',
})

print("\nการกระจายตัวของตัวแปรพฤติกรรม:")
print(f"  ความถี่    : {df['Freq_Group'].value_counts().to_dict()}")
print(f"  ช่องทาง   : {df['Channel_Group'].value_counts().to_dict()}")
print(f"  ประเภทเมนู: {df['Menu_Group'].value_counts().to_dict()}")

# ============================================================
# Step 3 : แปลงเป็น Long Format
# ============================================================
status_col = 'สถานภาพปัจจุบันที่เกี่ยวข้องกับมหาวิทยาลัยธรรมศาสตร์ ศูนย์รังสิต'
id_vars = ['ID', 'เพศ', 'อายุ', 'รายได้เฉลี่ยต่อเดือน (โดยประมาณ)', status_col,
           'Freq_Group', 'Channel_Group', 'Menu_Group']

df_long = pd.melt(df, id_vars=id_vars, value_vars=['add30', 'add50', 'add70'],
                  var_name='Bundle_Type', value_name='Will_Buy')
df_long = df_long.sort_values(by=['ID', 'Bundle_Type']).reset_index(drop=True)
print(f"\nจำนวนข้อสังเกต (Long Format): {len(df_long)}")

# ============================================================
# Step 4 : ยุบกลุ่มตัวแปร Demographic
# ============================================================
df_long['Age_Group'] = df_long['อายุ'].replace({
    'ต่ำกว่า 18 ปี': '22 ปี หรือต่ำกว่า', '18 - 22 ปี': '22 ปี หรือต่ำกว่า',
    '23 - 27 ปี': '23 ปีขึ้นไป', '28 - 35 ปี': '23 ปีขึ้นไป', '36 ปีขึ้นไป': '23 ปีขึ้นไป',
})
df_long['Income_Group'] = df_long['รายได้เฉลี่ยต่อเดือน (โดยประมาณ)'].replace({
    '20,001 - 30,000 บาท': 'มากกว่า 20,000 บาท',
    'มากกว่า 30,000 บาท':  'มากกว่า 20,000 บาท',
})
df_long['Status_Group'] = df_long[status_col].replace({
    'บุคคลทั่วไปที่อาศัย/ทำงานในบริเวณใกล้เคียง': 'บุคคลทั่วไป/คนทำงาน',
    'บุคลากร/อาจารย์': 'บุคคลทั่วไป/คนทำงาน',
})

# ============================================================
# Step 5 : โมเดล 2a — Pooled Logit (Categorical) + ตัวแปรพฤติกรรม
# ============================================================
print("\n" + "="*65)
print("โมเดล 2a: Logit Regression — Bundle_Type (Categorical) + ตัวแปรพฤติกรรม")
print("Base group = add30, หญิง, ≤22ปี, 5,001-10,000บาท, นักศึกษา, 1ครั้ง, In-store, Bundle")
print("="*65)

formula_2a = """
    Will_Buy ~ C(Bundle_Type, Treatment('add30')) +
               C(เพศ, Treatment('หญิง')) +
               C(Age_Group, Treatment('22 ปี หรือต่ำกว่า')) +
               C(Income_Group, Treatment('5,001 - 10,000 บาท')) +
               C(Status_Group, Treatment('นักศึกษา/บัณฑิตศึกษา')) +
               C(Freq_Group, Treatment('1 time')) +
               C(Channel_Group, Treatment('In-store')) +
               C(Menu_Group, Treatment('Bundle'))
"""

result_2a = smf.logit(formula_2a, data=df_long).fit(
    cov_type='cluster', cov_kwds={'groups': df_long['ID']}, disp=False
)
print(result_2a.summary())

print("\n" + "="*65)
print("โมเดล 2a: Marginal Effects (dy/dx)")
print("="*65)
me_2a = result_2a.get_margeff()
print(me_2a.summary())

# ============================================================
# Step 6 : โมเดล 2b — Pooled Logit (Continuous) สำหรับ WTP
# ============================================================
print("\n" + "="*65)
print("โมเดล 2b: Logit — Price_Increase (Continuous) + ตัวแปรพฤติกรรม")
print("สำหรับคำนวณ WTP เท่านั้น")
print("="*65)

df_long['Price_Increase'] = df_long['Bundle_Type'].map({'add30': 30, 'add50': 50, 'add70': 70})

formula_2b = """
    Will_Buy ~ Price_Increase +
               C(เพศ, Treatment('หญิง')) +
               C(Age_Group, Treatment('22 ปี หรือต่ำกว่า')) +
               C(Income_Group, Treatment('5,001 - 10,000 บาท')) +
               C(Status_Group, Treatment('นักศึกษา/บัณฑิตศึกษา')) +
               C(Freq_Group, Treatment('1 time')) +
               C(Channel_Group, Treatment('In-store')) +
               C(Menu_Group, Treatment('Bundle'))
"""

result_2b = smf.logit(formula_2b, data=df_long).fit(
    cov_type='cluster', cov_kwds={'groups': df_long['ID']}, disp=False
)
print(result_2b.summary())

print("\n" + "="*65)
print("โมเดล 2b: Marginal Effects (dy/dx)")
print("="*65)
me_2b = result_2b.get_margeff()
print(me_2b.summary())

# ============================================================
# Step 7 : คำนวณ WTP
# ============================================================
BASE_PRICE  = 199
intercept2  = result_2b.params['Intercept']
coef_price2 = result_2b.params['Price_Increase']
wtp2        = -intercept2 / coef_price2

print("\n" + "="*65)
print("WTP จากโมเดล 2b — กลุ่ม B (มีของแถม)")
print("="*65)
print(f"""
  Intercept                    : {intercept2:.4f}
  Coefficient (Price_Increase) : {coef_price2:.4f}
  WTP                          : {wtp2:.2f} บาท
  ราคาที่ยอมรับได้สูงสุด        : {BASE_PRICE + wtp2:.1f} บาท
""")

# ============================================================
# Step 8 : VIF Test
# ============================================================
print("\n" + "="*65)
print("VIF Test — ตรวจสอบ Multicollinearity")
print("="*65)

vif_data = dmatrix(
    "Price_Increase + C(Income_Group) + C(เพศ) + C(Age_Group) + C(Status_Group) + C(Freq_Group) + C(Channel_Group) + C(Menu_Group)",
    df_long, return_type='dataframe'
)
vif_df = pd.DataFrame({
    "Variable": vif_data.columns,
    "VIF":      [variance_inflation_factor(vif_data.values, i)
                 for i in range(vif_data.shape[1])]
})
print("\nVIF < 5 = ไม่มีปัญหา Multicollinearity")
print(vif_df[vif_df['Variable'] != 'Intercept'].sort_values(by='VIF', ascending=False).to_string(index=False))

# ============================================================
# Step 9 : Export ผลลัพธ์เป็น CSV
# ============================================================
import os

def to_csv(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"  ✅ {filename}")

def result_to_df(result):
    conf = result.conf_int()
    conf.columns = ['lower', 'upper']
    return pd.DataFrame({
        'Variable': result.params.index,
        'Coef':     result.params.values.round(4),
        'Std_Err':  result.bse.values.round(4),
        'z':        result.tvalues.values.round(3),
        'p_value':  result.pvalues.values.round(4),
        'CI_Lower': conf['lower'].values.round(4),
        'CI_Upper': conf['upper'].values.round(4),
    })

def me_to_df(me_result):
    sf = me_result.summary_frame()
    return pd.DataFrame({
        'Variable': sf.index,
        'dy/dx':    sf.iloc[:, 0].round(4),
        'Std_Err':  sf.iloc[:, 1].round(4),
        'z':        sf.iloc[:, 2].round(3),
        'p_value':  sf.iloc[:, 3].round(4),
        'CI_Lower': sf.iloc[:, 4].round(4),
        'CI_Upper': sf.iloc[:, 5].round(4),
    })

print("\n" + "="*65)
print("Export CSV — สมมติฐานข้อที่ 2 (เพิ่มตัวแปรพฤติกรรม)")
print("="*65)

to_csv(result_to_df(result_2a), 'H2B_Model2a_Logit_Categorical.csv')
to_csv(me_to_df(me_2a),         'H2B_Model2a_MarginalEffects.csv')
to_csv(result_to_df(result_2b), 'H2B_Model2b_Logit_Continuous.csv')
to_csv(me_to_df(me_2b),         'H2B_Model2b_MarginalEffects.csv')

to_csv(pd.DataFrame({
    'Item':  ['Intercept', 'Coef (Price_Increase)', 'WTP (บาท)', 'ราคาสูงสุดที่ยอมรับ (บาท)'],
    'Value': [round(intercept2,4), round(coef_price2,4), round(wtp2,2), round(BASE_PRICE+wtp2,1)]
}), 'H2B_WTP.csv')

to_csv(vif_df[vif_df['Variable'] != 'Intercept'].sort_values('VIF', ascending=False),
       'H2B_VIF.csv')

to_csv(pd.DataFrame([
    {'Metric': 'Pseudo R²',       'Model_2a': round(result_2a.prsquared,4), 'Model_2b': round(result_2b.prsquared,4)},
    {'Metric': 'Log-Likelihood',  'Model_2a': round(result_2a.llf,2),       'Model_2b': round(result_2b.llf,2)},
    {'Metric': 'LLR p-value',     'Model_2a': round(result_2a.llr_pvalue,6),'Model_2b': round(result_2b.llr_pvalue,6)},
    {'Metric': 'No. Observations','Model_2a': int(result_2a.nobs),          'Model_2b': int(result_2b.nobs)},
    {'Metric': 'WTP (บาท)',       'Model_2a': None,                         'Model_2b': round(wtp2,2)},
]), 'H2B_ModelSummary.csv')

print(f"\nบันทึกไฟล์ทั้งหมดที่โฟลเดอร์: {os.path.abspath('.')}")
print("โค้ดรันสำเร็จ!")
print("="*65)
