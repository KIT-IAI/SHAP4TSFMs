# Data Sources and Licensing

The code in this repository is licensed under the MIT License. The data files in this folder are subject to separate terms as described below.

---

## Weather Data — ERA5 via Copernicus Climate Change Service

Temperature, Global Horizontal Irradiance, wind speed, precipitation, and mean sea level pressure, aggregated to four NUTS-2 subdivisions of Baden-Württemberg, Germany:

| Administrative District | Code |
|------------------------|------|
| Stuttgart              | DE11 |
| Karlsruhe              | DE12 |
| Freiburg               | DE13 |
| Tübingen               | DE14 |

Units:

| Variable                    | Unit  |
|-----------------------------|-------|
| Air Temperature             | °C    |
| Global Horizontal Irradiance| W/m²  |
| Wind Speed                  | m/s   |
| Total Precipitation         | m     |
| Mean Sea Level Pressure     | Pa    |

**Source:** Copernicus Climate Change Service (C3S), European Centre for Medium-Range Weather Forecasts (ECMWF)
https://cds.climate.copernicus.eu

**License:** Copernicus Licence Agreement
https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf

**Required attribution:**
> Contains modified Copernicus Climate Change Service information [2024]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

---

## Electricity Load Data — ENTSO-E Transparency Platform

Actual total load for the TransnetBW control zone (Baden-Württemberg).

**Source:** ENTSO-E Transparency Platform
https://transparency.entsoe.eu

**License status:** Permission to republish has been requested from TransnetBW (email sent). Awaiting response.

> **Note:** This data may be subject to the ENTSO-E Terms & Conditions and/or TransnetBW's own data policies. Verify your right to use this data before redistribution.
