# streamlit_sales_dashboard.py
"""
Upgraded Streamlit Sales & Collections Dashboard (Gemini SDK integrated)
- Added CXO-level KPIs, executive visualizations, drill-downs, alerts, predictive features, improved UX/layout, and enhanced filters.
- Handles missing 'lat' and 'lon' columns gracefully for geographic map.
- Fixes NameError for 'customers' by moving sidebar filters after data loading.
- Fixes KeyError for 'AgingBucket' by reordering aging_buckets before compute_kpis.
- Uses google-generativeai SDK for Gemini calls.
- Install: pip install google-generativeai
- Set GEMINI_API_KEY in your environment or paste the key in the Gemini input box in-app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# plotting
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor

# PPTX (optional)
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    HAS_PPTX = True
except Exception:
    HAS_PPTX = False

st.set_page_config(page_title="Sales & Collections — CXO Dashboard", layout="wide", initial_sidebar_state="expanded")

# Set current date as specified
CURRENT_DATE = pd.Timestamp('2025-09-29')

# ---------------------- Helpers ----------------------
@st.cache_data
def load_excel_all_sheets(path_or_buffer):
    try:
        xl = pd.read_excel(path_or_buffer, sheet_name=None)
        return xl
    except Exception:
        try:
            return {"Sheet1": pd.read_csv(path_or_buffer)}
        except Exception:
            return None

def safe_parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def load_file_uploader(uploader, default_path=None):
    if uploader is not None:
        try:
            try:
                return pd.read_excel(uploader, sheet_name=None)
            except Exception:
                uploader.seek(0)
                return pd.read_csv(uploader)
        except Exception:
            return None
    if default_path and os.path.exists(default_path):
        try:
            return pd.read_excel(default_path, sheet_name=None)
        except Exception:
            return pd.read_csv(default_path)
    return None

def unwrap_excel(df_like, friendly_name='file'):
    if isinstance(df_like, dict):
        sheets = list(df_like.keys())
        if len(sheets) == 1:
            st.warning(f"{friendly_name} had one sheet ('{sheets[0]}'). Using it.")
            return df_like[sheets[0]]
        else:
            sel = st.sidebar.selectbox(f"Select sheet from {friendly_name}", options=sheets, index=0)
            st.info(f"Using sheet '{sel}' from {friendly_name}.")
            return df_like[sel]
    return df_like

def compute_kpis(merged, ref_date=CURRENT_DATE):
    if ref_date is None:
        ref_date = pd.Timestamp.now().normalize()
    total_invoiced = merged['Amount'].sum()
    total_collected = merged['Collected'].sum()
    outstanding = merged['Outstanding'].sum()
    merged['DaysOverdue'] = (ref_date - merged['DueDate']).dt.days.clip(lower=0)
    if merged['Outstanding'].sum() > 0:
        dso = (merged['DaysOverdue'] * merged['Outstanding']).sum() / merged['Outstanding'].sum()
    else:
        dso = 0
    collection_rate = total_collected / total_invoiced if total_invoiced else 0
    
    # Additional KPIs
    gross_ar = total_invoiced
    net_ar = outstanding
    overdue_30 = merged[merged['DaysOverdue'] >= 30]['Outstanding'].sum() / outstanding if outstanding else 0
    overdue_60 = merged[merged['DaysOverdue'] >= 60]['Outstanding'].sum() / outstanding if outstanding else 0
    overdue_90 = merged[merged['DaysOverdue'] >= 90]['Outstanding'].sum() / outstanding if outstanding else 0
    
    # Bad Debt Reserve
    risk_factors = {'Current': 0.01, '1-30': 0.05, '31-60': 0.1, '61-90': 0.2, '90+': 0.5}
    if 'AgingBucket' in merged.columns:
        merged['Reserve'] = merged.apply(lambda r: r['Outstanding'] * risk_factors.get(r['AgingBucket'], 0), axis=1)
        bad_debt_reserve = merged['Reserve'].sum()
    else:
        st.warning("AgingBucket column missing; bad debt reserve set to 0.")
        bad_debt_reserve = 0
    bad_debt_pct = (bad_debt_reserve / outstanding) * 100 if outstanding else 0
    
    # Cash at Risk
    cash_at_risk = bad_debt_reserve  # Simplified: use reserve as proxy if no aging buckets
    
    # Collections Efficiency
    collections_efficiency = collection_rate
    
    return dict(
        total_invoiced=total_invoiced, total_collected=total_collected, outstanding=outstanding, 
        dso=round(dso,1), collection_rate=round(collection_rate,3),
        gross_ar=gross_ar, net_ar=net_ar, overdue_30=overdue_30, overdue_60=overdue_60, overdue_90=overdue_90,
        bad_debt_reserve=bad_debt_reserve, bad_debt_pct=bad_debt_pct, cash_at_risk=cash_at_risk,
        collections_efficiency=collections_efficiency
    )

def aging_buckets(df, ref_date=CURRENT_DATE):
    if 'DueDate' not in df.columns:
        df['DueDate'] = df.get('InvoiceDate')
    df['DaysOverdue'] = (ref_date - df['DueDate']).dt.days
    bins = [-9999, 0, 30, 60, 90, 99999]
    labels = ['Current', '1-30', '31-60', '61-90', '90+']
    df['AgingBucket'] = pd.cut(df['DaysOverdue'], bins=bins, labels=labels, include_lowest=True)
    agg = df.groupby('AgingBucket')['Outstanding'].sum().reset_index()
    return agg, df

def run_isolation_forest(df, feature_cols=['Amount','Outstanding','DaysOverdue'], contamination=0.02):
    X = df[feature_cols].fillna(0).values
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)
    df['anomaly_score'] = clf.decision_function(X)
    df['is_anomaly'] = preds == -1
    return df.sort_values('anomaly_score')

def compute_risk_score_table(cust_df, merged_df):
    df = cust_df.copy()
    eps = 1e-9
    df['outstanding'] = df['outstanding'].fillna(0)
    total_inv = df.get('total_invoiced', df['outstanding']).sum()
    df['outstanding_share'] = df['outstanding'] / (total_inv + eps)

    def comp_overdue(g):
        if 'DueDate' in g.columns:
            days = (CURRENT_DATE - g['DueDate']).dt.days.clip(lower=0)
        else:
            days = (CURRENT_DATE - g['InvoiceDate']).dt.days.clip(lower=0)
        weighted = (days * (g['Outstanding']>0).astype(int)).sum()
        return weighted

    overdue = merged_df.groupby('CustomerID').apply(comp_overdue).rename('overdue_intensity').reset_index()
    df = df.merge(overdue, on='CustomerID', how='left')
    df['overdue_intensity'] = df['overdue_intensity'].fillna(0)
    if 'collection_rate' not in df.columns:
        df['collection_rate'] = (df.get('collected',0) / (df.get('total_invoiced', df['outstanding']) + eps)).fillna(0)

    df['z_outstanding'] = (df['outstanding'] - df['outstanding'].mean())/(df['outstanding'].std()+eps)
    df['z_overdue'] = (df['overdue_intensity'] - df['overdue_intensity'].mean())/(df['overdue_intensity'].std()+eps)
    df['z_collection'] = (df['collection_rate'].mean() - df['collection_rate'])/(df['collection_rate'].std()+eps)

    w_out, w_over, w_col = 0.5, 0.3, 0.2
    df['risk_score'] = (w_out * df['z_outstanding'] + w_over * df['z_overdue'] + w_col * df['z_collection']).fillna(0)
    df = df.sort_values('risk_score', ascending=False)

    def reason(r):
        parts = []
        if r['outstanding'] > 0:
            parts.append(f"Outstanding ₹{r['outstanding']:,.0f}")
        if r['overdue_intensity'] > 30:
            parts.append(f"Long overdue (intensity {int(r['overdue_intensity'])})")
        if r['collection_rate'] < 0.6:
            parts.append(f"Low collection rate {r['collection_rate']:.0%}")
        return '; '.join(parts) or 'No material issues'
    df['reason'] = df.apply(reason, axis=1)
    return df

def month_end_receivables_series(invoices_df, collections_df, start_date=None, end_date=None):
    inv = invoices_df.copy()
    coll = collections_df.copy()
    inv['InvoiceDate'] = pd.to_datetime(inv.get('InvoiceDate'), errors='coerce')
    coll['PaymentDate'] = pd.to_datetime(coll.get('PaymentDate'), errors='coerce')
    if start_date is None:
        try:
            start_date = min(inv['InvoiceDate'].min(), coll['PaymentDate'].min())
        except Exception:
            start_date = inv['InvoiceDate'].min() if not inv['InvoiceDate'].isna().all() else CURRENT_DATE - pd.DateOffset(months=12)
    if end_date is None:
        try:
            end_date = max(inv['InvoiceDate'].max(), coll['PaymentDate'].max())
        except Exception:
            end_date = CURRENT_DATE
    if pd.isna(start_date):
        start_date = CURRENT_DATE - pd.DateOffset(months=12)
    if pd.isna(end_date):
        end_date = CURRENT_DATE
    periods = pd.period_range(start=start_date.to_period('M'), end=end_date.to_period('M'), freq='M')
    rows = []
    for p in periods:
        me = p.to_timestamp('M')
        cum_inv = inv[inv['InvoiceDate'] <= me]['Amount'].sum()
        cum_coll = coll[coll['PaymentDate'] <= me]['Collection'].sum()
        rows.append({'MonthEnd': me, 'CumulativeInvoiced': cum_inv, 'CumulativeCollected': cum_coll, 'Receivables': cum_inv - cum_coll})
    return pd.DataFrame(rows)

def forecast_receivables_gb(monthly_df, n_periods=3):
    df = monthly_df.copy().sort_values('MonthEnd').reset_index(drop=True)
    df['y'] = df['Receivables']
    for lag in range(1,7):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['month'] = df['MonthEnd'].dt.month
    df['year'] = df['MonthEnd'].dt.year
    df_nonull = df.dropna().reset_index(drop=True)
    if df_nonull.shape[0] < 6:
        last = monthly_df['Receivables'].iloc[-1] if len(monthly_df) else 0
        last_month = monthly_df['MonthEnd'].max() if len(monthly_df) else CURRENT_DATE.to_period('M').to_timestamp()
        next_months = []
        for i in range(1, n_periods+1):
            nm = (last_month + pd.DateOffset(months=i))
            next_months.append({'MonthEnd': nm, 'ForecastReceivables': last})
        return pd.DataFrame(next_months)
    feat_cols = [c for c in df_nonull.columns if c.startswith('lag_')] + ['month','year']
    X, y = df_nonull[feat_cols], df_nonull['y']
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    cur = df_nonull.iloc[-1].copy()
    cur['MonthEnd'] = df_nonull['MonthEnd'].iloc[-1]
    cur['y'] = df_nonull['y'].iloc[-1]
    preds = []
    for i in range(1, n_periods+1):
        next_month_ts = (cur['MonthEnd'] + pd.DateOffset(months=1))
        feat = {}
        for lag in range(1,7):
            feat[f'lag_{lag}'] = cur['y'] if lag == 1 else cur.get(f'lag_{lag-1}', cur['y'])
        feat['month'] = next_month_ts.month
        feat['year'] = next_month_ts.year
        Xn = pd.DataFrame([feat])
        fval = model.predict(Xn)[0]
        preds.append({'MonthEnd': next_month_ts, 'ForecastReceivables': max(0, float(fval))})
        cur['y'] = fval
        for lag in range(6,1,-1):
            cur[f'lag_{lag}'] = cur.get(f'lag_{lag-1}', cur['y'])
        cur['lag_1'] = fval
        cur['MonthEnd'] = next_month_ts
    return pd.DataFrame(preds)

def concentration_metrics(cust_summary, top_n=3):
    total = cust_summary['total_invoiced'].sum() if 'total_invoiced' in cust_summary.columns else cust_summary['outstanding'].sum()
    if 'total_invoiced' in cust_summary.columns:
        csum = cust_summary.sort_values('total_invoiced', ascending=False).reset_index(drop=True)
        top_share = csum.head(top_n)['total_invoiced'].sum() / total if total else 0
        csum['share'] = csum['total_invoiced'] / total if total else 0
    else:
        csum = cust_summary.sort_values('outstanding', ascending=False).reset_index(drop=True)
        top_share = csum.head(top_n)['outstanding'].sum() / total if total else 0
        csum['share'] = csum['outstanding'] / total if total else 0
    hhi = (csum['share'] ** 2).sum()
    return {'top_n_share': top_share, 'hhi': hhi}

def build_analysis_input_md(kpis, monthly, mer_rcv, dso_trend, top_risks_df, forecast_df):
    lines = []
    lines.append("# analysis_input.md — Credit & Receivables Snapshot\n")
    lines.append("## KPIs")
    lines.append(f"- Total invoiced: ₹{kpis['total_invoiced']:,.0f}")
    lines.append(f"- Total collected: ₹{kpis['total_collected']:,.0f}")
    lines.append(f"- Outstanding receivables: ₹{kpis['outstanding']:,.0f}")
    lines.append(f"- DSO (current): {kpis['dso']} days")
    lines.append("\n## Trends (last months)")
    last_months = monthly.sort_values('InvoiceMonth', ascending=False).head(6).sort_values('InvoiceMonth')
    for r in last_months.itertuples():
        lines.append(f"- {r.InvoiceMonth.strftime('%Y-%m')}: invoiced ₹{r.invoiced:,.0f}, collected ₹{r.collected:,.0f}")
    lines.append("\n## Top risk customers (top 10 by risk_score)")
    for r in top_risks_df.head(10).itertuples():
        lines.append(f"- {r.CustomerName} ({r.CustomerID}): risk_score {r.risk_score:.3f}; {r.reason}")
    lines.append("\n## Receivables (month-end recent)")
    for r in mer_rcv.tail(6).itertuples():
        lines.append(f"- {r.MonthEnd.strftime('%Y-%m-%d')}: Receivables ₹{r.Receivables:,.0f}")
    lines.append("\n## DSO trend (recent)")
    for r in dso_trend.tail(6).itertuples():
        lines.append(f"- {r.MonthEnd.strftime('%Y-%m-%d')}: DSO {r.DSO:.1f} days")
    lines.append("\n## 3-month Forecast (Receivables)")
    for r in forecast_df.itertuples():
        lines.append(f"- {r.MonthEnd.strftime('%Y-%m-%d')}: Forecast Receivables ₹{r.ForecastReceivables:,.0f}")
    return "\n".join(lines)

def compute_cohort_analysis(merged):
    merged['InvoiceMonth'] = merged['InvoiceDate'].dt.to_period('M')
    merged['FirstInvoiceMonth'] = merged.groupby('CustomerID')['InvoiceMonth'].transform('min')
    cohort = merged.groupby(['FirstInvoiceMonth', 'InvoiceMonth'])['Amount'].sum().unstack()
    cohort_retention = cohort.divide(cohort.iloc[:,0], axis=0)
    return cohort_retention

def compute_ar_waterfall(mer_rcv, monthly):
    if len(mer_rcv) < 2:
        return pd.DataFrame()
    opening = mer_rcv['Receivables'].iloc[-2]
    invoiced = monthly['invoiced'].iloc[-1]
    collected = monthly['collected'].iloc[-1]
    closing = mer_rcv['Receivables'].iloc[-1]
    data = pd.DataFrame({
        'Category': ['Opening', '+ Invoiced', '- Collected', 'Closing'],
        'Amount': [opening, invoiced, -collected, closing]
    })
    data['Cumulative'] = data['Amount'].cumsum()
    return data

# ---------------------- Gemini SDK call ----------------------
def list_available_gemini_models(api_key):
    try:
        from google import generativeai as genai
        if api_key and str(api_key).strip():
            os.environ.setdefault("GEMINI_API_KEY", str(api_key).strip())
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        st.error(f"Failed to list Gemini models: {e}")
        return []

def call_gemini_api_generative(api_key, prompt, model="gemini-1.5-flash", temperature=0.2, max_tokens=600):
    try:
        from google import generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai SDK not installed. Run: pip install google-generativeai") from e

    if api_key and str(api_key).strip():
        os.environ.setdefault("GEMINI_API_KEY", str(api_key).strip())
    
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        available_models = list_available_gemini_models(api_key)
        if model not in available_models:
            st.warning(f"Model {model} not found or unsupported. Falling back to gemini-1.5-flash.")
            model = "gemini-1.5-flash" if "gemini-1.5-flash" in available_models else available_models[0] if available_models else None
            if not model:
                raise RuntimeError(f"No supported models available. Available models: {available_models}")
        
        response = genai.GenerativeModel(model).generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        if response.candidates and response.candidates[0].content:
            return response.candidates[0].content.parts[0].text
        return str(response)
    except Exception as e:
        available_models = list_available_gemini_models(api_key)
        raise RuntimeError(f"Gemini SDK call failed: {e}. Available models: {available_models}") from e

# ---------------------- Sidebar: inputs (before loading data) ----------------------
st.sidebar.header("Data inputs & options")
st.sidebar.markdown("Upload the three files (invoices, customers, collections) or place them next to the script.")

invoice_u = st.sidebar.file_uploader("Invoices file (Excel/CSV)", type=['xlsx','xls','csv'], key='invoices')
customers_u = st.sidebar.file_uploader("Customers master (Excel/CSV)", type=['xlsx','xls','csv'], key='customers')
collections_u = st.sidebar.file_uploader("Collections/Payments (Excel/CSV)", type=['xlsx','xls','csv'], key='collections')

DEFAULT_INVOICES = "./invoices.xlsx"
DEFAULT_CUSTOMERS = "./customers.xlsx"
DEFAULT_COLLECTIONS = "./collections.xlsx"

with st.sidebar.expander("Column mapping (if headers differ)"):
    inv_customer_col = st.text_input("Invoice -> CustomerID", value='CustomerID')
    inv_invoiceid_col = st.text_input("Invoice -> InvoiceID", value='InvoiceID')
    inv_date_col = st.text_input("Invoice -> InvoiceDate", value='InvoiceDate')
    inv_due_col = st.text_input("Invoice -> DueDate", value='DueDate')
    inv_amount_col = st.text_input("Invoice -> Amount", value='Amount')

    cust_id_col = st.text_input("Customer -> CustomerID", value='CustomerID')
    cust_name_col = st.text_input("Customer -> CustomerName", value='CustomerName')
    cust_sales_col = st.text_input("Customer -> Salesperson", value='salesperson')
    cust_div_col = st.text_input("Customer -> Division", value='Division')
    cust_cat_col = st.text_input("Customer -> Category", value='Category')
    cust_lat_col = st.text_input("Customer -> Latitude", value='lat')
    cust_lon_col = st.text_input("Customer -> Longitude", value='lon')

    coll_customer_col = st.text_input("Collections -> CustomerID", value='CustomerID')
    coll_invoice_col = st.text_input("Collections -> InvoiceId", value='InvoiceId')
    coll_payment_col = st.text_input("Collections -> PaymentDate", value='PaymentDate')
    coll_collection_col = st.text_input("Collections -> Collection amount", value='Collection')

st.sidebar.markdown("---")
contamination = st.sidebar.slider("Anomaly contamination", 0.0, 0.1, 0.02, 0.005)
alert_threshold = st.sidebar.number_input("Alert threshold (₹ outstanding)", value=500000, step=10000)
st.sidebar.markdown("---")
date_range = st.sidebar.date_input("Invoice date range", [CURRENT_DATE - timedelta(days=365), CURRENT_DATE])

# ---------------------- Load & unwrap files ----------------------
invoices = load_file_uploader(invoice_u, DEFAULT_INVOICES)
customers = load_file_uploader(customers_u, DEFAULT_CUSTOMERS)
collections = load_file_uploader(collections_u, DEFAULT_COLLECTIONS)

invoices = unwrap_excel(invoices, 'Invoices')
customers = unwrap_excel(customers, 'Customers')
collections = unwrap_excel(collections, 'Collections')

if invoices is None or customers is None or collections is None:
    st.warning("Missing inputs: upload invoices/customers/collections or place default files in working folder.")
    st.stop()

# ---------------------- Cleaning & merge ----------------------
invoices = invoices.rename(columns=lambda x: str(x).strip())
invoices = invoices.rename(columns={inv_customer_col: 'CustomerID', inv_invoiceid_col: 'InvoiceID', inv_date_col:'InvoiceDate', inv_due_col:'DueDate', inv_amount_col:'Amount'})
invoices = safe_parse_dates(invoices, ['InvoiceDate','DueDate'])

customers = customers.rename(columns=lambda x: str(x).strip())
customers = customers.rename(columns={cust_id_col:'CustomerID', cust_name_col:'CustomerName', cust_sales_col:'salesperson', cust_div_col:'Division', cust_cat_col:'Category', cust_lat_col:'lat', cust_lon_col:'lon'})

collections = collections.rename(columns=lambda x: str(x).strip())
collections = collections.rename(columns={coll_customer_col:'CustomerID', coll_invoice_col:'InvoiceId', coll_payment_col:'PaymentDate', coll_collection_col:'Collection'})
collections = safe_parse_dates(collections, ['PaymentDate'])

collections = collections.rename(columns={'InvoiceId':'InvoiceID'})
coll_agg = collections.groupby('InvoiceID', dropna=False).agg({'Collection':'sum'}).reset_index()

# Conditionally include lat and lon in merge
merge_columns = ['CustomerID', 'CustomerName', 'salesperson', 'Division', 'Category']
if 'lat' in customers.columns and 'lon' in customers.columns:
    merge_columns.extend(['lat', 'lon'])
else:
    st.warning("Latitude ('lat') and/or Longitude ('lon') columns missing in customers data. Geographic map will not be displayed. Add these columns to your customers file for geographic visualization.")

merged = invoices.merge(customers[merge_columns], on='CustomerID', how='left')
merged = merged.merge(coll_agg, on='InvoiceID', how='left')
merged['Collected'] = merged['Collection'].fillna(0)
merged['Outstanding'] = (merged.get('Amount',0) - merged['Collected']).clip(lower=0)
merged['InvoiceDate'] = pd.to_datetime(merged.get('InvoiceDate'), errors='coerce')
merged['DueDate'] = pd.to_datetime(merged.get('DueDate'), errors='coerce')

# Apply date filter
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
merged = merged[(merged['InvoiceDate'] >= start_date) & (merged['InvoiceDate'] <= end_date + pd.Timedelta(days=1))]

# ---------------------- Sidebar: filters (after data loading) ----------------------
st.sidebar.markdown("## Filters")
period_filter = st.sidebar.selectbox("Period", options=['MTD', 'QTD', 'YTD', 'Custom'], index=3)
customer_segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    options=customers['Category'].unique() if customers is not None and 'Category' in customers.columns else [],
    help="Select customer segments to filter data."
)
country_filter = st.sidebar.multiselect(
    "Country",
    options=customers['Country'].unique() if customers is not None and 'Country' in customers.columns else [],
    help="Select countries to filter data (requires 'Country' column)."
)
sales_rep_filter = st.sidebar.multiselect(
    "Sales Rep",
    options=customers['salesperson'].unique() if customers is not None and 'salesperson' in customers.columns else [],
    help="Select sales reps to filter data."
)
product_line_filter = st.sidebar.multiselect(
    "Product Line",
    options=merged['ProductLine'].unique() if 'ProductLine' in merged.columns else [],
    help="Select product lines to filter data (requires 'ProductLine' column)."
)
aging_threshold = st.sidebar.slider("Aging Bucket Threshold", 0, 180, 90)
currency = st.sidebar.selectbox("Currency", options=['INR', 'USD'], index=0)
compare_prior = st.sidebar.checkbox("Compare to prior period", value=False)

# Apply enhanced filters
if customer_segment_filter:
    merged = merged[merged['Category'].isin(customer_segment_filter)]
if sales_rep_filter:
    merged = merged[merged['salesperson'].isin(sales_rep_filter)]
if country_filter and 'Country' in merged.columns:
    merged = merged[merged['Country'].isin(country_filter)]
if product_line_filter and 'ProductLine' in merged.columns:
    merged = merged[merged['ProductLine'].isin(product_line_filter)]

# ---------------------- Derived metrics ----------------------
# Compute aging buckets first to ensure AgingBucket column exists
aging_agg, merged = aging_buckets(merged, ref_date=CURRENT_DATE)

# Compute KPIs after aging buckets
kpis = compute_kpis(merged)

sp_summary = merged.groupby('salesperson').agg(total_invoiced=('Amount','sum'), collected=('Collected','sum'), outstanding=('Outstanding','sum')).reset_index()
sp_summary['collection_rate'] = (sp_summary['collected'] / sp_summary['total_invoiced']).fillna(0)

merged['InvoiceMonth'] = merged['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
monthly = merged.groupby('InvoiceMonth').agg(invoiced=('Amount','sum'), collected=('Collected','sum')).reset_index()
monthly['collection_efficiency'] = (monthly['collected'] / monthly['invoiced']).replace([np.inf, -np.inf], np.nan).fillna(0)

# Month-end receivables & DSO trend
mer_rcv = month_end_receivables_series(invoices, collections, start_date=start_date, end_date=end_date)
dso_rows = []
for _, r in mer_rcv.iterrows():
    me = r['MonthEnd']
    inv_up_to = invoices[invoices['InvoiceDate'] <= me].copy()
    if inv_up_to.empty:
        dso_rows.append({'MonthEnd': me, 'Receivables': r['Receivables'], 'DSO': 0})
        continue
    coll_up_to = collections[collections['PaymentDate'] <= me].copy()
    coll_by_inv = coll_up_to.groupby('InvoiceID')['Collection'].sum()
    inv_up_to['Collected_cum'] = inv_up_to['InvoiceID'].map(coll_by_inv).fillna(0).values
    inv_up_to['Outstanding_at_me'] = (inv_up_to['Amount'] - inv_up_to['Collected_cum']).clip(lower=0)
    if 'DueDate' in inv_up_to.columns:
        days_overdue = (me - inv_up_to['DueDate']).dt.days.clip(lower=0)
    else:
        days_overdue = (me - inv_up_to['InvoiceDate']).dt.days.clip(lower=0)
    if inv_up_to['Outstanding_at_me'].sum() > 0:
        dso_me = (days_overdue * inv_up_to['Outstanding_at_me']).sum() / inv_up_to['Outstanding_at_me'].sum()
    else:
        dso_me = 0
    dso_rows.append({'MonthEnd': me, 'Receivables': r['Receivables'], 'DSO': dso_me})
dso_trend = pd.DataFrame(dso_rows)

# Anomalies
try:
    merged_af = merged.copy()
    merged_af['DaysOverdue'] = (CURRENT_DATE - merged_af['DueDate']).dt.days.clip(lower=0)
    merged_af = run_isolation_forest(merged_af, feature_cols=['Amount','Outstanding','DaysOverdue'], contamination=contamination)
    anomalies = merged_af[merged_af['is_anomaly']].sort_values('anomaly_score')
except Exception:
    anomalies = pd.DataFrame()

# Churn & concentration
cust_summary = merged.groupby(['CustomerID','CustomerName','salesperson','Division','Category']).agg(total_invoiced=('Amount','sum'), collected=('Collected','sum'), outstanding=('Outstanding','sum'), invoice_count=('InvoiceID','nunique')).reset_index()
cust_summary['collection_rate'] = (cust_summary['collected']/cust_summary['total_invoiced']).fillna(0)
cust_summary['churn_risk_score'] = (1 - cust_summary['collection_rate']) * (cust_summary['outstanding'] / (cust_summary['total_invoiced']+1))
conc = concentration_metrics(cust_summary, top_n=3)
sp_monthly = merged.groupby(['salesperson','InvoiceMonth']).agg(invoiced=('Amount','sum'), collected=('Collected','sum')).reset_index()
sp_monthly['collection_efficiency'] = (sp_monthly['collected'] / sp_monthly['invoiced']).replace([np.inf, -np.inf], np.nan).fillna(0)

# Cohort
cohort_retention = compute_cohort_analysis(merged)

# Waterfall
ar_waterfall = compute_ar_waterfall(mer_rcv, monthly)

# Forecast
forecast_df = forecast_receivables_gb(mer_rcv, n_periods=3)

# Risk scoring
risk_tbl = compute_risk_score_table(cust_summary, merged)

# Alerts
high_risk_customers = risk_tbl[risk_tbl['risk_score'] > 1.0]
concentration_alert = conc['top_n_share'] > 0.2
if concentration_alert:
    st.sidebar.warning("Concentration threshold exceeded (>20% AR in top 3)")

# ---------------------- UI ----------------------
st.markdown("<h1>Sales & Collections — CXO Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Strategic insights into receivables, collections, risks, and forecasts.")

# Top bar: selectors
col_company, col_period, col_currency = st.columns(3)
with col_company:
    company_selector = st.selectbox("Company", options=['All'], index=0)
with col_period:
    period_selector = st.selectbox("Period", options=['MTD', 'QTD', 'YTD', 'Custom'], index=3)
with col_currency:
    currency_toggle = st.selectbox("Currency", options=['INR', 'USD'], index=0)

# CXO KPIs row
st.subheader("CXO-Level KPIs")
kpi_cols = st.columns(6)
with kpi_cols[0]:
    st.metric("Gross Receivables", f"₹{kpis['gross_ar']:,.0f}", delta=0, help="Total invoiced amount")
    st.markdown("**Action**: Monitor invoicing trends.")
with kpi_cols[1]:
    st.metric("Net Receivables", f"₹{kpis['net_ar']:,.0f}", delta=0, help="Outstanding after collections")
    st.markdown("**Action**: Prioritize collections.")
with kpi_cols[2]:
    color = "red" if kpis['dso'] > 60 else "amber" if kpis['dso'] > 30 else "green"
    st.metric("DSO", f"{kpis['dso']} days", delta=0, delta_color='inverse', help="Days Sales Outstanding")
    st.markdown(f"<span style='color:{color}'>**Action**: Reduce DSO.</span>", unsafe_allow_html=True)
with kpi_cols[3]:
    color = "red" if kpis['overdue_90'] > 0.2 else "amber" if kpis['overdue_90'] > 0.1 else "green"
    st.metric("% Overdue >=90", f"{kpis['overdue_90']:.1%}", delta=0, delta_color='inverse', help="Percentage of AR overdue 90+ days")
    st.markdown(f"<span style='color:{color}'>**Action**: Escalate overdue accounts.</span>", unsafe_allow_html=True)
with kpi_cols[4]:
    color = "red" if conc['top_n_share'] > 0.3 else "amber" if conc['top_n_share'] > 0.2 else "green"
    st.metric("AR Concentration (Top 3 %)", f"{conc['top_n_share']*100:.1f}%", delta=0, delta_color='inverse', help="% of AR in top 3 customers")
    st.markdown(f"<span style='color:{color}'>**Action**: Diversify customer base.</span>", unsafe_allow_html=True)
with kpi_cols[5]:
    color = "red" if kpis['bad_debt_pct'] > 5 else "amber" if kpis['bad_debt_pct'] > 2 else "green"
    st.metric("Bad Debt %", f"{kpis['bad_debt_pct']:.1f}%", delta=0, delta_color='inverse', help="Bad debt reserve as % of AR")
    st.markdown(f"<span style='color:{color}'>**Action**: Review reserve policy.</span>", unsafe_allow_html=True)

# More KPIs in next row
kpi_cols2 = st.columns(6)
with kpi_cols2[0]:
    color = "red" if kpis['cash_at_risk'] > kpis['net_ar'] * 0.2 else "amber" if kpis['cash_at_risk'] > kpis['net_ar'] * 0.1 else "green"
    st.metric("Cash at Risk", f"₹{kpis['cash_at_risk']:,.0f}", delta=0, delta_color='inverse', help="Overdue * default probability")
    st.markdown(f"<span style='color:{color}'>**Action**: Mitigate high-risk accounts.</span>", unsafe_allow_html=True)
with kpi_cols2[1]:
    color = "green" if kpis['collections_efficiency'] > 0.9 else "amber" if kpis['collections_efficiency'] > 0.7 else "red"
    st.metric("Collections Efficiency", f"{kpis['collections_efficiency']:.1%}", delta=0, help="% collected vs billed")
    st.markdown(f"<span style='color:{color}'>**Action**: Improve collection processes.</span>", unsafe_allow_html=True)

# Two-column body
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Strategic Charts")
    # AR Trend with forecast
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=mer_rcv['MonthEnd'], y=mer_rcv['Receivables'], mode='lines', name='Receivables'))
    fig_trend.add_trace(go.Scatter(x=forecast_df['MonthEnd'], y=forecast_df['ForecastReceivables'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig_trend.update_layout(title="AR Trend with Forecast", xaxis_title="Month", yaxis_title="Receivables (₹)")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("**Implication**: Monitor trend to ensure liquidity.")

    # Aging Heatmap
    heatmap_df = merged.pivot_table(index='CustomerName', columns='AgingBucket', values='Outstanding', aggfunc='sum').fillna(0)
    fig_heatmap = px.imshow(heatmap_df, aspect='auto', color_continuous_scale='Reds')
    fig_heatmap.update_layout(title="Aging Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("**Implication**: Focus on high-value overdue buckets.")

    # Pareto
    pareto_df = cust_summary.sort_values('outstanding', ascending=False).head(10)
    pareto_df['cum_pct'] = pareto_df['outstanding'].cumsum() / pareto_df['outstanding'].sum() * 100
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(x=pareto_df['CustomerName'], y=pareto_df['outstanding'], name='Outstanding'))
    fig_pareto.add_trace(go.Scatter(x=pareto_df['CustomerName'], y=pareto_df['cum_pct'], name='Cum %', mode='lines+markers', yaxis='y2'))
    fig_pareto.update_layout(title="Customer Concentration (Pareto)", yaxis2=dict(overlaying='y', side='right'))
    st.plotly_chart(fig_pareto, use_container_width=True)
    st.markdown("**Implication**: Reduce reliance on top customers.")

    # Waterfall
    if not ar_waterfall.empty:
        fig_waterfall = go.Figure(go.Waterfall(
            name="AR Movement", orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=ar_waterfall['Category'], y=ar_waterfall['Amount']
        ))
        fig_waterfall.update_layout(title="AR Waterfall")
        st.plotly_chart(fig_waterfall, use_container_width=True)
        st.markdown("**Implication**: Understand AR movement drivers.")

    # Geo Map
    if 'lat' in merged.columns and 'lon' in merged.columns:
        fig_geo = px.scatter_mapbox(merged.groupby('CustomerID').agg({'lat':'first', 'lon':'first', 'Outstanding':'sum'}).reset_index(),
                                    lat='lat', lon='lon', size='Outstanding', hover_name='CustomerID',
                                    mapbox_style="open-street-map")
        fig_geo.update_layout(title="AR by Geography")
        st.plotly_chart(fig_geo, use_container_width=True)
        st.markdown("**Implication**: Assess regional risk exposure.")
    else:
        st.info("Geographic map not displayed due to missing 'lat' and 'lon' columns.")

with right_col:
    st.subheader("Actionable Lists")
    # Top Debtors
    top_debtors = cust_summary.sort_values('outstanding', ascending=False).head(25)[['CustomerName', 'outstanding', 'salesperson', 'collection_rate']]
    st.dataframe(top_debtors.style.format({'outstanding':'{0:,.0f}', 'collection_rate':'{:.1%}'}), use_container_width=True)

    # Alerts & Playbooks
    st.subheader("Alerts & Recommendations")
    for _, row in high_risk_customers.iterrows():
        color = "red" if row['risk_score'] > 1.5 else "amber"
        st.markdown(f"<span style='color:{color}'>High risk: {row['CustomerName']} - {row['reason']}. Action: Escalate to sales.</span>", unsafe_allow_html=True)

    # One-click actions
    if st.button("Send Collection Emails to Top 10"):
        st.info("Emails sent (placeholder).")

# Drill-down tabs
st.markdown("## Drill-Down Views")
drill_tabs = st.tabs(["Customer Profile", "Collector Performance", "Aging Movements"])

with drill_tabs[0]:
    selected_cust = st.selectbox("Select Customer", options=merged['CustomerName'].unique())
    cust_invoices = merged[merged['CustomerName'] == selected_cust]
    st.dataframe(cust_invoices)

with drill_tabs[1]:
    st.dataframe(sp_summary.style.format({'total_invoiced':'{0:,.0f}', 'collected':'{0:,.0f}', 'outstanding':'{0:,.0f}', 'collection_rate':'{:.1%}'}))

with drill_tabs[2]:
    aging_movements = merged.groupby(['InvoiceMonth', 'AgingBucket'])['Outstanding'].sum().unstack().fillna(0)
    st.dataframe(aging_movements)

# Advanced analysis
st.markdown("## Advanced Analysis")
tab1, tab2, tab3 = st.tabs(["Dimension Analysis", "AI: Gemini Analysis", "Forecast & Exports"])

with tab1:
    st.subheader("Dimension Analysis (Salesperson / Division / Category)")
    for d in ['salesperson', 'Division', 'Category']:
        st.markdown(f"**Detailed Analysis By {d}**")
        if d in merged.columns:
            t = merged.groupby(d).apply(lambda g: pd.Series({
                'total_invoiced': g['Amount'].sum(),
                'collected': g['Collected'].sum(),
                'outstanding': g['Outstanding'].sum(),
                'collection_rate': g['Collected'].sum() / g['Amount'].sum() if g['Amount'].sum() else 0,
                'dso': (g['DaysOverdue'] * g['Outstanding']).sum() / g['Outstanding'].sum() if g['Outstanding'].sum() else 0,
                'invoice_count': g['InvoiceID'].nunique()
            })).reset_index()
            t = t.sort_values('outstanding', ascending=False).head(25)
            st.dataframe(t.style.format({'total_invoiced':'{0:,.0f}', 'collected':'{0:,.0f}', 'outstanding':'{0:,.0f}', 'collection_rate':'{:.1%}', 'dso':'{:.1f}'}), use_container_width=True)
            
            fig_dim = px.bar(t.head(10), x=d, y='outstanding', color='dso', title=f'Outstanding by {d} (Top 10)')
            fig_dim.update_layout(template='plotly_white')
            st.plotly_chart(fig_dim, use_container_width=True)
            
            fig_coll = px.line(t.head(10), x=d, y='collection_rate', title=f'Collection Rate by {d} (Top 10)')
            fig_coll.update_layout(template='plotly_white')
            st.plotly_chart(fig_coll, use_container_width=True)
        else:
            st.write(f"No {d} dimension present in data.")

with tab2:
    st.subheader("Gemini / LLM-driven analysis (CFO-ready)")
    gemini_key = st.text_input("Gemini API Key (optional)", type='password')
    available_models = list_available_gemini_models(gemini_key)
    selected_model = st.selectbox("Select Gemini Model", options=available_models if available_models else ["gemini-1.5-flash"], index=0)
    run_gemini = st.button("Generate CFO analysis (Gemini)", key='run_gemini')
    analysis_md = build_analysis_input_md(kpis, monthly, mer_rcv, dso_trend, risk_tbl, forecast_df)
    st.download_button("Download analysis_input.md", data=analysis_md, file_name="analysis_input.md", mime="text/markdown", key='download_analysis_md')
    st.text_area("analysis_input.md (preview)", value=analysis_md, height=300)
    if run_gemini:
        if gemini_key:
            user_prompt = f"""You are a senior credit & receivables analyst.
based on the receivables, sales and collection data you have analysed, provide the following

1) Executive Summary (Overall receivables, Trend in DSO, High risk and Low risk features, Any abnormal trend in ageing)
2) Top 10 Risky Customers (1–2 lines each: why risky, quant evidence, suggested action)
3) Pareto analysis of receivables, riskiness in the view of industry/ customer category concentration, default probability, 
4) Sales person analysis ( Additional sales bought in, colleciton efficiency)
5)Aread need key attention

Rules:
- Base everything strictly on the provided numbers; do not invent figures.
- Be concise (< 600 words), CFO-ready.
- Use Indian business English.

Here is the analysis_input.md content:
{analysis_md}
"""
            try:
                with st.spinner("Calling Gemini..."):
                    gemini_resp = call_gemini_api_generative(gemini_key, user_prompt, model=selected_model)
                st.subheader("Gemini response")
                st.text_area("Gemini Output", value=str(gemini_resp), height=450)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
        else:
            st.info("No Gemini key provided — showing local executive summary (fallback).")
            exec_summary = []
            try:
                recent = monthly.sort_values('InvoiceMonth', ascending=False).head(3)['invoiced'].sum()
                prev3 = monthly.sort_values('InvoiceMonth', ascending=False).iloc[3:6]['invoiced'].sum()
                pct_change = (recent - prev3)/ (prev3+1e-9)
                exec_summary.append(f"Invoiced changed {pct_change:.1%} vs prior 3 months.")
            except Exception:
                pass
            exec_summary.append(f"Outstanding ₹{kpis['outstanding']:,.0f}, DSO {kpis['dso']} days.")
            exec_summary.append(f"Top risk customers: {', '.join(risk_tbl['CustomerName'].head(5).tolist())}")
            st.markdown("**Executive summary (local fallback)**")
            for b in exec_summary:
                st.markdown("- " + b)

with tab3:
    st.subheader("Forecast & Exports (extra)")
    st.dataframe(forecast_df.style.format({'ForecastReceivables':'{0:,.0f}'}), use_container_width=True)
    st.download_button("Download forecast CSV (extra)", data=forecast_df.to_csv(index=False).encode('utf-8'), file_name='receivables_forecast_extra.csv', mime='text/csv', key='download_forecast_tab3')
    st.download_button("Export Dashboard Snapshot (PDF)", data="placeholder.pdf", file_name="dashboard_snapshot.pdf")  # Placeholder for PDF export

# Footer
st.markdown("---")
st.markdown(f"Last refresh: {CURRENT_DATE.strftime('%Y-%m-%d %H:%M')} | Data source: Uploaded files | Contact: AR Team")

with st.expander("How to use this dashboard"):
    st.markdown("- Use filters to focus on a salesperson, division, or segment.")
    st.markdown("- Add 'lat' and 'lon' columns to customers file for geographic visualization.")
    st.markdown("- For production, store API keys as environment variables and never commit them to source control.")