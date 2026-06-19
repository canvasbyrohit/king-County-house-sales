# Data Dictionary — kc_house_cleaned.csv
# B141 Data Mining — King County House Sales

## Original Features (21)

| Column | Type | Description |
|---|---|---|
| id | str | Unique property identifier |
| date | datetime | Sale transaction date |
| price | int | Sale price in USD (TARGET) |
| bedrooms | int | Number of bedrooms (corrected: 33→3) |
| bathrooms | float | Number of bathrooms (incl. half-baths) |
| sqft_living | int | Living area in square feet |
| sqft_lot | int | Lot area in square feet |
| floors | float | Number of floors |
| waterfront | int | Binary — 1 if waterfront property |
| view | int | View quality rating (0–4) |
| condition | int | Property condition rating (1–5) |
| grade | int | King County quality grade (1–13) |
| sqft_above | int | Living area above ground (sqft) |
| sqft_basement | int | Basement area (sqft; 0 = no basement) |
| yr_built | int | Year property was built |
| yr_renovated | int | Year last renovated (0 = never) |
| zipcode | int | US Postal zipcode |
| lat | float | Latitude coordinate |
| long | float | Longitude coordinate |
| sqft_living15 | int | Avg living area of 15 nearest neighbours |
| sqft_lot15 | int | Avg lot area of 15 nearest neighbours |

## Engineered Features — Phase 3 (9)

| Column | Type | Description |
|---|---|---|
| house_age | int | Years since built at time of sale |
| renovated | int | Binary — 1 if ever renovated |
| yrs_since_reno | int | Years since last renovation (0 = never) |
| sale_month | int | Month of sale (1–12) |
| sale_year | int | Year of sale |
| basement_flag | int | Binary — 1 if property has a basement |
| price_per_sqft | float | price / sqft_living (EDA only — exclude from model) |
| living_ratio | float | sqft_living / sqft_living15 |
| lot_ratio | float | sqft_lot / sqft_lot15 |

## Engineered Features — Phase 5 (6)

| Column | Type | Description |
|---|---|---|
| grade_sqft | int | grade × sqft_living (quality-size interaction) |
| bath_bed_ratio | float | bathrooms / (bedrooms + 1) |
| total_rooms | int | bedrooms + int(bathrooms) |
| high_grade | int | Binary — 1 if grade >= 10 (luxury tier) |
| log_sqft | float | log1p(sqft_living) |
| log_lot | float | log1p(sqft_lot) |

## Model Target

| Column | Transformation | Description |
|---|---|---|
| price | log1p(price) | Sale price — log-transformed for modelling |
