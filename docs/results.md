# Results (300 Monte Carlo runs, 5 km segments)

## 2022 weather

| plan | race time (without nights) | arrival (local) | lowest SoC, nominal car | runs below 10 % SoC |
|---|---|---|---|---|
| constant speed 73.5 km/h | 45 h 35 min | 27 Aug, 08:35 | 10.4 % | 48.7 % |
| optimized | 45 h 18 min | 27 Aug, 08:17 | 10.0 % | 51.3 % |
| risk-targeted, alpha = 0.948 | 47 h 31 min | 27 Aug, 10:31 | 46.6 % | 4.7 % |

## Back-test over 10 weather years

| year | constant speed | optimized gain | optimized risk | alpha | risk after scaling | time cost |
|---|---|---|---|---|---|---|
| 2016 | 72.9 km/h | 3 min | 50.0 % | 0.946 | 5.0 % | +141 min |
| 2017 | 77.1 km/h | 4 min | 49.7 % | 0.917 | 4.7 % | +213 min |
| 2018 | 76.3 km/h | 27 min | 51.3 % | 0.939 | 4.7 % | +151 min |
| 2019 | 74.5 km/h | 9 min | 49.7 % | 0.938 | 4.7 % | +159 min |
| 2020 | 80.5 km/h | 10 min | 50.0 % | 0.934 | 5.0 % | +160 min |
| 2021 | 75.1 km/h | 5 min | 49.7 % | 0.948 | 4.7 % | +131 min |
| 2022 | 73.5 km/h | 17 min | 51.3 % | 0.948 | 4.7 % | +134 min |
| 2023 | 79.9 km/h | 3 min | 49.7 % | 0.929 | 4.7 % | +174 min |
| 2024 | 79.7 km/h | 15 min | 50.0 % | 0.913 | 5.0 % | +215 min |
| 2025 | 80.1 km/h | 18 min | 50.3 % | 0.921 | 5.0 % | +193 min |
