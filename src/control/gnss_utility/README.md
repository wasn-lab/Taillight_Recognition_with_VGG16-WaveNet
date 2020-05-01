# GNSS Utility
GNSS utility supports 

- TWD97 TM2 (E, N) <-> WGS84 (lat, lon)     
 - (m, m) <-> (deg, deg)
- WGS84 (lat, lon, h) <-> ECEF (X, Y, Z)    
 - (deg, deg, m) <-> (m, m, m)
- ECEF (X, Y, Z) <-> ENU (E, N, U), given ref. coord. (lat0, lon0, h0)
 - (m, m, m) <-> (m, m, m)


**WGS84 = World Geodetic System 1984**

**TWD97 TM2 = Taiwan datum 1997, 2-degree transverse Mercator**

**ECEF = earth-centered earth-fixed coordinate system**

**ENU = East, North, Up coordinates, a local Earth based coordinate system**

## Test
- mkdir build
- cd build
- cmake ..
- make
- ./gnss_test

## Test Result

###TEST 1 : NMEA\_RMC -> TWD97 (E, N, Heading)

[RMC Input\] $GPRMC,090110.25,A,2446.622573,N,12102.585780,E,0.049,16.78,211299,4.26,W,F,S*60

\[RMCtoTWD97\] (E, N, Heading) = (254357.959841, 2741083.486217, 16.780000)


###TEST 2 : NMEA\_GGA -> TWD97 (E,N) -> WGS84 (lat,lon) -> TWD97 (E,N)


\[GGA Input\] $GPGGA,090110.25,2446.622573,N,12102.585780,E,5,22,0.0,112.311,M,19.583,M,1.0,0088*4A

\[GGAtoTWD97\] (E, N) = (254357.959841, 2741083.486217)

\[TWD97toWGS84\] (lat, lon) = (24.777043, 121.043096)

\[WGS84toTWD97\] (E, N) = (254357.958482, 2741083.486555)

###TEST 3 : WGS84 (lat, lon, h) -> ECEF (X, Y, Z) -> ENU (E, N, U) -> ECEF (X, Y, Z) -> WGS84 (lat, lon, h)

\[WGS84 Input\] (lat, lon, h) = (34.000000, -117.333569, 251.702000)

\[WGS84toECEF\] (X, Y, Z) = (-2430601.823904, -4702442.705311, 3546587.357796)

\[ECEFtoENU\] (E, N, U) = (0.000000, 0.000000, 0.000000)

- Ref. Coord. (lat0, lon0, h0) = (34.000000, -117.333569, 251.702000)

\[ENUtoECEF\] (X, Y, Z) = (-2430601.823904, -4702442.705311, 3546587.357796)

- Ref. Coord. (lat0, lon0, h0) = (34.000000, -117.333569, 251.702000)

\[ECEFtoWGS84\] (lat, lon, h) = (34.000000, -117.333569, 251.702000)

###TEST 4 : WGS84(lat, lon, h) -> ENU(E, N, U)
\[WGS84 Input\] (lat, lon, h) = (24.775580, 121.042485, 118.771000)

\[WGS84toENU\](E, N, U) = (-49.181739, -36.321930, -0.557293)

- Ref. Coord. (lat0, lon0, h0) = (24.775908, 121.042972, 119.328000)

## Usage
include gnss\_utility.h and gnss\_utility.cpp in your project and call API function similarly as in gnss\_test.cpp.

## Formula Reference
- https://github.com/yychen/twd97 (TWD97<->WGS84)
- https://gist.github.com/govert/1b373696c9a27ff4c72a (WGS84<->ECEF<->ENU)

## Validation Reference

- http://www.sunriver.com.tw/taiwanmap/grid_tm2_convert.php#a03 (TWD97<->WGS84)