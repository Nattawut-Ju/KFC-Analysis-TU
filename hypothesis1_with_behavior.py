"""
============================================================
สมมติฐานข้อที่ 1 — โค้ดฉบับสมบูรณ์ (เพิ่ม 3 ตัวแปรพฤติกรรม)
Pooled Logit Model with Clustered Standard Errors (by ID)
ใช้กลุ่ม A : +20, +40, +60 บาท (ไม่มีของแถม)

ตัวแปร Control เพิ่มเติม:
  - ความถี่ในการซื้อ KFC ต่อเดือน
  - ช่องทางหลักในการซื้อ KFC
  - ประเภทเมนูที่ซื้อปกติ
============================================================
วิธีใช้:
1. วางไฟล์ Excel ไว้ในโฟลเดอร์เดียวกับไฟล์ .py นี้
2. รันคำสั่ง: python hypothesis1_with_behavior.py
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

# เปลี่ยนชื่อคอลัมน์ราคา
rename_dict = {
    'หากราคาปรับขึ้นเป็น 219 บาท (+20 บาท) คุณจะยังซื้ออยู่ไหม?': '20',
    'หากราคาปรับขึ้นเป็น 239 บาท (+40 บาท) คุณจะยังซื้ออยู่ไหม?': '40',
    'หากราคาปรับขึ้นเป็น 259 บาท (+60 บาท) คุณจะยังซื้ออยู่ไหม':  '60',
}
df.rename(columns=rename_dict, inplace=True)

for col in ['20', '40', '60']:
    df[col] = df[col].astype(str).str.strip().map({'ซื้อ': 1, 'ไม่ซื้อ': 0})
df = df.dropna(subset=['20', '40', '60'])
for col in ['20', '40', '60']:
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
# Delivery มีแค่ 1 คน → รวมกับ App/Online
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
status_col   = 'สถานภาพปัจจุบันที่เกี่ยวข้องกับมหาวิทยาลัยธรรมศาสตร์ ศูนย์รังสิต'
id_vars = ['ID', 'เพศ', 'อายุ', 'รายได้เฉลี่ยต่อเดือน (โดยประมาณ)', status_col,
           'Freq_Group', 'Channel_Group', 'Menu_Group']

df_long = pd.melt(df, id_vars=id_vars, value_vars=['20', '40', '60'],
                  var_name='Price_Increase', value_name='Will_Buy')
df_long['Price_Increase'] = df_long['Price_Increase'].astype(int)
df_long = df_long.sort_values(by=['ID', 'Price_Increase']).reset_index(drop=True)
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
df_long['Gender'] = df_long['เพศ']

# ============================================================
# Step 5 : โมเดล 1 — เพิ่ม 3 ตัวแปรพฤติกรรม
# ============================================================
print("\n" + "="*65)
print("โมเดล 1: Logit Regression + ตัวแปรพฤติกรรม")
print("Base: หญิง, ≤22ปี, 5,001-10,000บาท, นักศึกษา, 1ครั้ง, In-store, Bundle")
print("="*65)

formula1 = """
    Will_Buy ~ Price_Increase +
               C(Gender, Treatment('หญิง')) +
               C(Age_Group, Treatment('22 ปี หรือต่ำกว่า')) +
               C(Income_Group, Treatment('5,001 - 10,000 บาท')) +
               C(Status_Group, Treatment('นักศึกษา/บัณฑิตศึกษา')) +
               C(Freq_Group, Treatment('1 time')) +
               C(Channel_Group, Treatment('In-store')) +
               C(Menu_Group, Treatment('Bundle'))
"""

result1 = smf.logit(formula1, data=df_long).fit(
    cov_type='cluster', cov_kwds={'groups': df_long['ID']}, disp=False
)
print(result1.summary())

print("\n" + "="*65)
print("โมเดล 1: Marginal Effects (dy/dx)")
print("="*65)
me1 = result1.get_margeff()
print(me1.summary())

# ============================================================
# Step 6 : คำนวณ WTP
# ============================================================
BASE_PRICE  = 199
intercept1  = result1.params['Intercept']
coef_price1 = result1.params['Price_Increase']
wtp1        = -intercept1 / coef_price1

print("\n" + "="*65)
print("WTP จากโมเดล 1")
print("="*65)
print(f"""
  Intercept                    : {intercept1:.4f}
  Coefficient (Price_Increase) : {coef_price1:.4f}
  WTP                          : {wtp1:.2f} บาท
  ราคาที่ยอมรับได้สูงสุด        : {BASE_PRICE + wtp1:.1f} บาท
""")

# ============================================================
# Step 7 : โมเดล 2 — เพิ่ม Interaction (Price × Income) + ตัวแปรพฤติกรรม
# ============================================================
print("\n" + "="*65)
print("โมเดล 2: Logit + Interaction (Price × Income) + ตัวแปรพฤติกรรม")
print("="*65)

formula2 = """
    Will_Buy ~ Price_Increase * C(Income_Group, Treatment('5,001 - 10,000 บาท')) +
               C(Gender, Treatment('หญิง')) +
               C(Age_Group, Treatment('22 ปี หรือต่ำกว่า')) +
               C(Status_Group, Treatment('นักศึกษา/บัณฑิตศึกษา')) +
               C(Freq_Group, Treatment('1 time')) +
               C(Channel_Group, Treatment('In-store')) +
               C(Menu_Group, Treatment('Bundle'))
"""

result2 = smf.logit(formula2, data=df_long).fit(
    cov_type='cluster', cov_kwds={'groups': df_long['ID']}, disp=False
)
print(result2.summary())

print("\n" + "="*65)
print("โมเดล 2: Marginal Effects (dy/dx)")
print("="*65)
me2 = result2.get_margeff()
print(me2.summary())

# ============================================================
# Step 8 : VIF Test
# ============================================================
print("\n" + "="*65)
print("VIF Test — ตรวจสอบ Multicollinearity")
print("="*65)

vif_data = dmatrix(
    "Price_Increase + C(Income_Group) + C(Gender) + C(Age_Group) + C(Status_Group) + C(Freq_Group) + C(Channel_Group) + C(Menu_Group)",
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
# Step 9 : สรุปเปรียบเทียบ 2 โมเดล
# ============================================================
print("\n" + "="*65)
print("สรุปเปรียบเทียบโมเดล 1 และโมเดล 2")
print("="*65)
print(f"""
                          โมเดล 1      โมเดล 2
  Pseudo R²            : {result1.prsquared:.4f}       {result2.prsquared:.4f}
  Log-Likelihood       : {result1.llf:.2f}     {result2.llf:.2f}
  LLR p-value          : {result1.llr_pvalue:.2e}   {result2.llr_pvalue:.2e}
  No. Observations     : {int(result1.nobs)}          {int(result2.nobs)}

  Coef (Price_Increase): {result1.params['Price_Increase']:.4f}       {result2.params['Price_Increase']:.4f}
  p-value (Price)      : {result1.pvalues['Price_Increase']:.4f}       {result2.pvalues['Price_Increase']:.4f}
  WTP เฉลี่ย           : {wtp1:.2f} บาท
""")

# ============================================================
# Step 10 : Export ผลลัพธ์เป็น CSV
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
print("Export CSV — สมมติฐานข้อที่ 1 (เพิ่มตัวแปรพฤติกรรม)")
print("="*65)

to_csv(result_to_df(result1), 'H1B_Model1_Logit.csv')
to_csv(me_to_df(me1),         'H1B_Model1_MarginalEffects.csv')

to_csv(pd.DataFrame({
    'Item':  ['Intercept', 'Coef (Price_Increase)', 'WTP (บาท)', 'ราคาสูงสุดที่ยอมรับ (บาท)'],
    'Value': [round(intercept1,4), round(coef_price1,4), round(wtp1,2), round(BASE_PRICE+wtp1,1)]
}), 'H1B_WTP.csv')

to_csv(result_to_df(result2), 'H1B_Model2_Logit_Interaction.csv')
to_csv(me_to_df(me2),         'H1B_Model2_MarginalEffects.csv')

to_csv(vif_df[vif_df['Variable'] != 'Intercept'].sort_values('VIF', ascending=False),
       'H1B_VIF.csv')

to_csv(pd.DataFrame([
    {'Metric': 'Pseudo R²',            'Model_1': round(result1.prsquared,4),           'Model_2': round(result2.prsquared,4)},
    {'Metric': 'Log-Likelihood',       'Model_1': round(result1.llf,2),                 'Model_2': round(result2.llf,2)},
    {'Metric': 'LLR p-value',          'Model_1': round(result1.llr_pvalue,6),           'Model_2': round(result2.llr_pvalue,6)},
    {'Metric': 'No. Observations',     'Model_1': int(result1.nobs),                    'Model_2': int(result2.nobs)},
    {'Metric': 'Coef (Price_Increase)','Model_1': round(result1.params['Price_Increase'],4), 'Model_2': round(result2.params['Price_Increase'],4)},
    {'Metric': 'p-value (Price)',       'Model_1': round(result1.pvalues['Price_Increase'],4),'Model_2': round(result2.pvalues['Price_Increase'],4)},
    {'Metric': 'WTP (บาท)',            'Model_1': round(wtp1,2),                        'Model_2': None},
]), 'H1B_ModelSummary.csv')

print(f"\nบันทึกไฟล์ทั้งหมดที่โฟลเดอร์: {os.path.abspath('.')}")
print("โค้ดรันสำเร็จ!")
