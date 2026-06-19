# =============================================================================
# B141 DATA MINING — King County House Sales
# Phase 4: Exploratory Data Analysis & Visualisations
# =============================================================================

# ── CELL 1 : Setup ────────────────────────────────────────────────────────────
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

PALETTE = ['#2E4057','#048A81','#54C6EB','#EFA00B','#E05263','#6B4226','#8338EC']
BLUE, TEAL, CORAL, GOLD = '#2E4057','#048A81','#E05263','#EFA00B'
plt.rcParams.update({'figure.dpi':150,'axes.titlesize':13,'axes.labelsize':11,
                     'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False})

df = pd.read_csv('kc_house_cleaned.csv', parse_dates=['date'])
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── CELL 2 : Dataset Overview ─────────────────────────────────────────────────
"""
DATASET OVERVIEW
Key statistical properties of the cleaned dataset.
"""
print("\n── NUMERICAL SUMMARY ────────────────────────────────────────")
print(df[['price','sqft_living','bedrooms','bathrooms','grade','house_age']
         ].describe(percentiles=[.05,.25,.5,.75,.95]).round(1).T.to_string())

print("\n── CATEGORICAL / BINARY SUMMARY ─────────────────────────────")
for col in ['waterfront','view','condition','grade','renovated','basement_flag']:
    vc = df[col].value_counts(normalize=True).mul(100).round(1)
    print(f"  {col:15s}: {vc.to_dict()}")

# ── CELL 3 : Fig 1 — Price Distribution ──────────────────────────────────────
"""
FIG 1 — TARGET VARIABLE: PRICE DISTRIBUTION (Univariate)
What    : Histogram of raw vs log-transformed sale price.
Why     : Identifies right skew that violates linear regression assumptions.
Business: Majority of sales cluster below $1M; luxury segment is thin but pulls the mean up.
Decision: Apply log1p transformation to price for all linear models.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 1 — Target Variable: Sale Price Distribution', fontsize=14, fontweight='bold')
axes[0].hist(df['price']/1e6, bins=80, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].axvline(df['price'].mean()/1e6, color=CORAL, lw=2, label=f"Mean ${df['price'].mean()/1e6:.2f}M")
axes[0].axvline(df['price'].median()/1e6, color=GOLD, lw=2, ls='--', label=f"Median ${df['price'].median()/1e6:.2f}M")
axes[0].set_xlabel('Price (USD millions)'); axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Raw Price  |  Skewness: {df["price"].skew():.2f}'); axes[0].legend()
axes[1].hist(np.log1p(df['price']), bins=80, color=TEAL, edgecolor='white', alpha=0.85)
axes[1].set_xlabel('log(Price + 1)'); axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Log-Transformed  |  Skewness: {np.log1p(df["price"]).skew():.2f}')
plt.tight_layout(); plt.savefig('fig01_price_distribution.png', bbox_inches='tight'); plt.show()

# ── CELL 4 : Fig 2 — Grade vs Price ──────────────────────────────────────────
"""
FIG 2 — PROPERTY GRADE vs SALE PRICE (Bivariate — Ordinal × Continuous)
What    : Bar chart of median price per grade + box plots showing spread.
Why     : Grade is a holistic quality score assigned by King County assessors.
Business: Grade 11–13 properties command 3–5× the price of grade 6–7 homes.
Decision: Grade should be a top-priority variable in pricing models.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 2 — Property Grade vs Sale Price', fontsize=14, fontweight='bold')
grade_med = df.groupby('grade')['price'].median()
axes[0].bar(grade_med.index, grade_med.values/1e6, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Grade (1–13)'); axes[0].set_ylabel('Median Price (USD millions)')
axes[0].set_title('Median Sale Price by Grade')
bp = axes[1].boxplot([df[df['grade']==g]['price'].values/1e6 for g in sorted(df['grade'].unique())],
                     labels=sorted(df['grade'].unique()), patch_artist=True,
                     medianprops=dict(color='white',lw=2))
for p in bp['boxes']: p.set_facecolor(TEAL); p.set_alpha(0.7)
axes[1].set_xlabel('Grade'); axes[1].set_ylabel('Price (USD millions)')
axes[1].set_title('Price Distribution by Grade')
plt.tight_layout(); plt.savefig('fig02_grade_vs_price.png', bbox_inches='tight'); plt.show()

# ── CELL 5 : Fig 3 — Correlation Heatmap ─────────────────────────────────────
"""
FIG 3 — CORRELATION HEATMAP (Multivariate)
What    : Lower-triangle Pearson correlation matrix for key numeric features.
Why     : Reveals multicollinearity risks and strength of feature–target relationships.
Business: sqft_living, grade, bathrooms, and lat dominate price correlation.
Decision: Monitor sqft_above + sqft_living for multicollinearity before modelling.
"""
corr_cols = ['price','sqft_living','sqft_lot','bedrooms','bathrooms','floors',
             'waterfront','view','condition','grade','house_age','renovated',
             'basement_flag','living_ratio','lat']
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, annot_kws={'size':8}, cbar_kws={'shrink':0.8})
ax.set_title('Fig 3 — Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('fig03_correlation_heatmap.png', bbox_inches='tight'); plt.show()

# ── CELL 6 : Fig 4 — sqft_living vs Price ────────────────────────────────────
"""
FIG 4 — LIVING AREA vs SALE PRICE (Bivariate Scatter + Regression)
What    : Scatter of sqft_living vs price, coloured by grade; log-log regression overlay.
Why     : Living area is the most intuitive physical driver of property value.
Business: Log-log R² = 0.49 — size alone explains ~49% of price variance.
Decision: Combine with grade for compound premium pricing model.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 4 — Living Area vs Sale Price', fontsize=14, fontweight='bold')
sc = axes[0].scatter(df['sqft_living'], df['price']/1e6, c=df['grade'], cmap='viridis', alpha=0.3, s=8)
plt.colorbar(sc, ax=axes[0], label='Grade')
axes[0].set_xlabel('Living Area (sqft)'); axes[0].set_ylabel('Price (USD millions)')
axes[0].set_title('Price vs Living Area (coloured by Grade)')
axes[1].scatter(np.log1p(df['sqft_living']), np.log1p(df['price']), color=TEAL, alpha=0.2, s=8)
m, b, r, _, _ = stats.linregress(np.log1p(df['sqft_living']), np.log1p(df['price']))
x_l = np.linspace(np.log1p(df['sqft_living']).min(), np.log1p(df['sqft_living']).max(), 100)
axes[1].plot(x_l, m*x_l+b, color=CORAL, lw=2, label=f'R²={r**2:.3f}')
axes[1].set_xlabel('log(sqft_living)'); axes[1].set_ylabel('log(price)')
axes[1].set_title('Log-Log Scatter (Linearised Relationship)'); axes[1].legend()
plt.tight_layout(); plt.savefig('fig04_sqft_vs_price.png', bbox_inches='tight'); plt.show()

# ── CELL 7 : Fig 5 — Geographic Price Map ────────────────────────────────────
"""
FIG 5 — GEOSPATIAL PRICE DISTRIBUTION (Spatial Analysis)
What    : Lat/Long scatter coloured by absolute price and price-per-sqft.
Why     : Location (lat/long) explains spatial price stratification.
Business: North-east corridor (Bellevue/Medina) commands highest prices and sqft rates.
Decision: Use lat, long, zipcode as location proxies in all models.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fig 5 — Geospatial Price Distribution Across King County', fontsize=14, fontweight='bold')
sc1 = axes[0].scatter(df['long'], df['lat'], c=df['price']/1e6, cmap='YlOrRd', alpha=0.4, s=5)
plt.colorbar(sc1, ax=axes[0], label='Price (USD millions)')
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude'); axes[0].set_title('Sale Price by Location')
sc2 = axes[1].scatter(df['long'], df['lat'], c=df['price_per_sqft'], cmap='plasma', alpha=0.4, s=5, vmin=100, vmax=800)
plt.colorbar(sc2, ax=axes[1], label='Price per sqft (USD)')
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude'); axes[1].set_title('Price per sqft by Location')
plt.tight_layout(); plt.savefig('fig05_geographic_price.png', bbox_inches='tight'); plt.show()

# ── CELL 8 : Fig 6 — Waterfront Premium ──────────────────────────────────────
"""
FIG 6 — WATERFRONT PRICE PREMIUM (Bivariate — Binary × Continuous)
What    : Overlapping histograms + median bar chart for waterfront vs non-waterfront.
Why     : Tests Hypothesis H2 — waterfront properties should command a premium.
Business: Waterfront homes sell at ~3× the median of non-waterfront properties.
Decision: Waterfront flag is a critical pricing input; lenders should adjust LTV ratios.
"""
wf = {0:'Non-Waterfront', 1:'Waterfront'}
colors_wf = [BLUE, TEAL]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 6 — Waterfront Property Price Premium', fontsize=14, fontweight='bold')
for i, (val, label) in enumerate(wf.items()):
    subset = df[df['waterfront']==val]['price']/1e6
    axes[0].hist(subset, bins=50, alpha=0.65, label=f'{label} (n={len(subset):,})',
                 color=colors_wf[i], edgecolor='white')
axes[0].set_xlabel('Price (USD millions)'); axes[0].legend()
axes[0].set_title('Price Distribution: Waterfront vs Non-Waterfront')
wf_stats = df.groupby('waterfront')['price'].agg(['median','count'])
wf_stats.index = ['Non-Waterfront','Waterfront']
axes[1].bar(wf_stats.index, wf_stats['median']/1e6, color=colors_wf, edgecolor='white', alpha=0.85)
for i, (idx, row) in enumerate(wf_stats.iterrows()):
    axes[1].text(i, row['median']/1e6+0.02, f"${row['median']/1e6:.2f}M", ha='center', fontsize=10)
prem = (wf_stats.loc['Waterfront','median']/wf_stats.loc['Non-Waterfront','median']-1)*100
axes[1].set_title(f'Median Price | Waterfront Premium: +{prem:.0f}%')
axes[1].set_ylabel('Median Price (USD millions)')
plt.tight_layout(); plt.savefig('fig06_waterfront_premium.png', bbox_inches='tight'); plt.show()

# ── CELL 9 : Fig 7 — Monthly Trend ───────────────────────────────────────────
"""
FIG 7 — MONTHLY SALES VOLUME & PRICE TREND (Time Series)
What    : Monthly transaction volume and median price over 12-month window.
Why     : Identifies seasonality patterns in the real estate market.
Business: Peak sales in Apr–Jul; prices stable suggesting volume-led not price-led cycles.
Decision: Seasonal marketing campaigns should front-load spring/summer listings.
"""
monthly = df.groupby(df['date'].dt.to_period('M')).agg(
    count=('price','count'), median_price=('price','median')).reset_index()
monthly['date_dt'] = monthly['date'].dt.to_timestamp()
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle('Fig 7 — Monthly Sales Volume & Median Price Trend', fontsize=14, fontweight='bold')
axes[0].fill_between(monthly['date_dt'], monthly['count'], alpha=0.4, color=BLUE)
axes[0].plot(monthly['date_dt'], monthly['count'], color=BLUE, lw=2)
axes[0].set_ylabel('Number of Sales'); axes[0].set_title('Monthly Transaction Volume')
axes[1].fill_between(monthly['date_dt'], monthly['median_price']/1e6, alpha=0.4, color=TEAL)
axes[1].plot(monthly['date_dt'], monthly['median_price']/1e6, color=TEAL, lw=2)
axes[1].set_ylabel('Median Price (USD millions)'); axes[1].set_xlabel('Month')
axes[1].set_title('Monthly Median Sale Price')
plt.tight_layout(); plt.savefig('fig07_monthly_trend.png', bbox_inches='tight'); plt.show()

# ── CELL 10 : Fig 8 — Age & Renovation ───────────────────────────────────────
"""
FIG 8 — HOUSE AGE, RENOVATION & PRICE (Bivariate + Engineered Feature)
What    : Scatter of house_age vs price split by renovation flag; median by age group.
Why     : Tests whether renovation reverses age-related depreciation.
Business: Renovated homes hold value better; 50–100yr renovated homes price near new builds.
Decision: Flag renovation history as a value-add in property listings.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 8 — House Age, Renovation & Price', fontsize=14, fontweight='bold')
for val, label, col in [(0,'Never Renovated',BLUE),(1,'Renovated',TEAL)]:
    sub = df[df['renovated']==val]
    axes[0].scatter(sub['house_age'], sub['price']/1e6, alpha=0.2, s=6,
                    label=f'{label} (n={len(sub):,})', color=col)
axes[0].set_xlabel('House Age (years)'); axes[0].set_ylabel('Price (USD millions)')
axes[0].set_title('House Age vs Price by Renovation Status'); axes[0].legend()
age_bins = pd.cut(df['house_age'], bins=[0,10,20,30,50,70,100,120],
                  labels=['0-10','11-20','21-30','31-50','51-70','71-100','100+'])
age_price = df.groupby(age_bins, observed=True)['price'].median()/1e6
axes[1].bar(age_price.index, age_price.values, color=PALETTE[:len(age_price)], edgecolor='white', alpha=0.85)
axes[1].set_xlabel('House Age Group (years)'); axes[1].set_ylabel('Median Price (USD millions)')
axes[1].set_title('Median Price by Age Group')
plt.tight_layout(); plt.savefig('fig08_age_renovation_price.png', bbox_inches='tight'); plt.show()

# ── CELL 11 : Fig 9 — Feature Correlation Bar ────────────────────────────────
"""
FIG 9 — FEATURE CORRELATION RANKING (Multivariate Summary)
What    : Horizontal bar chart of each feature's Pearson correlation with price.
Why     : Provides a quick visual feature importance proxy before modelling.
Business: sqft_living (0.70), grade (0.67), bathrooms (0.53) are the top 3 drivers.
Decision: Prioritise these in stakeholder dashboards and model interpretation.
"""
feat_cols = ['sqft_living','grade','bathrooms','view','sqft_living15','lat',
             'waterfront','bedrooms','floors','condition','house_age',
             'renovated','basement_flag','living_ratio','sqft_lot']
corr_price = df[feat_cols+['price']].corr()['price'].drop('price').sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [CORAL if x<0 else TEAL for x in corr_price.values]
bars = ax.barh(corr_price.index, corr_price.values, color=colors_bar, edgecolor='white', alpha=0.85)
ax.axvline(0, color='grey', lw=0.8, ls='--')
for bar, val in zip(bars, corr_price.values):
    ax.text(val+(0.005 if val>=0 else -0.005), bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val>=0 else 'right', fontsize=9)
ax.set_xlabel('Pearson Correlation with Price')
ax.set_title('Fig 9 — Feature Correlation with Sale Price (Ranked)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('fig09_feature_correlation.png', bbox_inches='tight'); plt.show()

# ── CELL 12 : Fig 10 — Zipcode Price Analysis ────────────────────────────────
"""
FIG 10 — MEDIAN PRICE BY ZIPCODE — TOP & BOTTOM 15 (Distribution Analysis)
What    : Ranked horizontal bar charts of median sale price for zipcodes.
Why     : Location is the single largest uncontrollable price factor.
Business: Zipcodes 98039, 98004, 98040 are premium; 98002, 98023 are entry-level.
Decision: Investors target bottom-15 zipcodes for affordable-stock development.
"""
zip_stats = df.groupby('zipcode')['price'].agg(['median','count']).reset_index()
zip_stats.columns = ['zipcode','median_price','count']
zip_stats = zip_stats.sort_values('median_price', ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fig 10 — Median Sale Price by Zipcode (Top & Bottom 15)', fontsize=14, fontweight='bold')
top15 = zip_stats.head(15); bottom15 = zip_stats.tail(15)
axes[0].barh(top15['zipcode'].astype(str), top15['median_price']/1e6, color=TEAL, edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Median Price (USD millions)'); axes[0].set_title('Top 15 Highest-Value Zipcodes'); axes[0].invert_yaxis()
axes[1].barh(bottom15['zipcode'].astype(str), bottom15['median_price']/1e6, color=BLUE, edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Median Price (USD millions)'); axes[1].set_title('Bottom 15 Lowest-Value Zipcodes'); axes[1].invert_yaxis()
plt.tight_layout(); plt.savefig('fig10_zipcode_price.png', bbox_inches='tight'); plt.show()

# ── CELL 13 : Fig 11 — Bedroom & Bathroom Distributions ─────────────────────
"""
FIG 11 — BEDROOM & BATHROOM COUNT DISTRIBUTIONS (Univariate)
What    : Bar charts of bedroom and bathroom frequency.
Why     : Reveals the typical property configuration in King County.
Business: 3-bedroom is modal (dominant); half-bathrooms common (en-suite culture).
Decision: 3–4 bedroom stock should be primary inventory focus for agents.
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 11 — Bedroom & Bathroom Count Distributions', fontsize=14, fontweight='bold')
bed_c = df['bedrooms'].value_counts().sort_index()
axes[0].bar(bed_c.index, bed_c.values, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Number of Bedrooms'); axes[0].set_ylabel('Count'); axes[0].set_title('Bedroom Distribution')
bath_c = df['bathrooms'].value_counts().sort_index()
axes[1].bar(bath_c.index, bath_c.values, color=TEAL, edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Number of Bathrooms'); axes[1].set_ylabel('Count'); axes[1].set_title('Bathroom Distribution')
plt.tight_layout(); plt.savefig('fig11_bed_bath_dist.png', bbox_inches='tight'); plt.show()

# ── CELL 14 : Fig 12 — Bedrooms × Grade Heatmap ──────────────────────────────
"""
FIG 12 — MEDIAN PRICE HEATMAP: BEDROOMS × GRADE (Multivariate)
What    : Pivot table heatmap of median price across bedroom count and grade combinations.
Why     : Exposes interaction effects not visible in single-feature analysis.
Business: A 4-bed grade-10 home ($1.2M) vs 4-bed grade-7 home ($0.42M) — grade doubles value.
Decision: Grade-led renovation (not just room additions) delivers highest ROI.
"""
pivot = df[df['bedrooms'].between(1,6)].pivot_table(
    values='price', index='bedrooms', columns='grade', aggfunc='median') / 1e6
pivot = pivot[[c for c in pivot.columns if pivot[c].notna().sum() > 2]]
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, cbar_kws={'label':'Median Price (USD millions)'})
ax.set_title('Fig 12 — Median Price Heatmap: Bedrooms × Grade', fontsize=13, fontweight='bold')
ax.set_xlabel('Grade'); ax.set_ylabel('Bedrooms')
plt.tight_layout(); plt.savefig('fig12_bedrooms_grade_heatmap.png', bbox_inches='tight'); plt.show()

print("\n✅ Phase 4 EDA Complete — 12 visualisations generated")
