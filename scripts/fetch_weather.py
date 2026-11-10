"""Download ERA5 weather along the route for race week (21-31 August).

Usage: python scripts/fetch_weather.py 2016 2025   (first and last year)
"""
import sys
import time

from solarstrat import fetch_era5, load_route


def main() -> None:
    first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) >= 3 else (2016, 2025)
    route = load_route()
    for year in range(first, last + 1):
        print("saved", fetch_era5(route, year))
        time.sleep(1.0)  # be polite to the free API


if __name__ == "__main__":
    main()
