#include <stdio.h>	//for printf()
#include <stdlib.h>	//for atof()
#include <string.h>	//for strtok()
#include <math.h>
#include <iostream>

#include "gnss_utility/gnss_utility.h"

namespace gnss_utility
{

gnss::gnss():
PI(3.14159265359),
DEG_PER_RAD(180.0/PI),
a(6378.137),
b(6356.752314245),
a_m (a*1000),
b_m (b*1000),
f(1 / 298.257222101),
k0(0.9999),
N0(0),
E0(250.000),
lon0(DegreesToRadians(121)), //radians(121)
lon0pkm(DegreesToRadians(119)), //radians(119)
n(f / (2-f)),
A(a / (1+n) * (1 + pow(n,2)/4.0 + pow(n,4)/64.0)),
alpha1(n/2 - 2*pow(n,2)/3.0 + 5*pow(n,3)/16.0),
alpha2(13*pow(n,2)/48.0 - 3*pow(n,3)/5.0),
alpha3(61*pow(n,3)/240.0),
beta1(n/2 - 2*pow(n,2)/3.0 + 37*pow(n,3)/96.0),
beta2(pow(n,2)/48.0 + pow(n,3)/15.0),
beta3(17*pow(n,3)/480.0),
delta1(2*n - 2*pow(n,2)/3.0 - 2*pow(n,3)),
delta2(7*pow(n,2)/3.0 - 8*pow(n,3)/5.0),
delta3(56*pow(n,3)/15.0),
e_sq(f*(2-f))
{
}

void gnss::WGS84toECEF(double lat, double lon, double h, double* X, double* Y, double* Z)
{
  // Convert to radians in notation consistent with the paper:
  double lambda = DegreesToRadians(lat);  // to rad
  double phi = DegreesToRadians(lon);  // to rad
  double s = sin(lambda);
  double N = a_m / sqrt(1 - e_sq * s * s);

  double sin_lambda = sin(lambda);
  double cos_lambda = cos(lambda);
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);

  *X = (h + N) * cos_lambda * cos_phi;
  *Y = (h + N) * cos_lambda * sin_phi;
  *Z = (h + (1 - e_sq) * N) * sin_lambda;
}

void gnss::ECEFtoWGS84(double X, double Y, double Z, double* lat, double* lon, double* h)
{
  double eps = e_sq / (1.0 - e_sq);
  double p = sqrt(X * X + Y * Y);
  double q = atan2((Z * a_m), (p * b_m));
  double sin_q = sin(q);
  double cos_q = cos(q);
  double sin_q_3 = sin_q * sin_q * sin_q;
  double cos_q_3 = cos_q * cos_q * cos_q;
  double phi = atan2((Z + eps * b_m * sin_q_3), (p - e_sq * a_m * cos_q_3));
  double lambda = atan2(Y, X);

  double v = a_m / sqrt(1.0 - e_sq * sin(phi) * sin(phi));
  *h = (p / cos(phi)) - v;

  *lat = phi * DEG_PER_RAD; // to deg
  *lon = lambda * DEG_PER_RAD; // to deg
}

void gnss::ECEFtoENU(double X, double Y, double Z, double lat0, double lon0, double h0, double* E, double* N, double* U)
{
  // Convert to radians in notation consistent with the paper:
  double lambda = DegreesToRadians(lat0); //to rad
  double phi = DegreesToRadians(lon0); //to rad
  double s = sin(lambda);
  double N_tmp = a_m / sqrt(1 - e_sq * s * s);

  double sin_lambda = sin(lambda);
  double cos_lambda = cos(lambda);
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);

  double x0 = (h0 + N_tmp) * cos_lambda * cos_phi;
  double y0 = (h0 + N_tmp) * cos_lambda * sin_phi;
  double z0 = (h0 + (1 - e_sq) * N_tmp) * sin_lambda;

  double xd = X - x0;
  double yd = Y - y0;
  double zd = Z - z0;

  // This is the matrix multiplication
  *E = -sin_phi * xd + cos_phi * yd;
  *N = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd;
  *U = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd;
}

void gnss::ENUtoECEF(double E, double N, double U, double lat0, double lon0, double h0, double* X, double* Y, double* Z)
{
  // Convert to radians in notation consistent with the paper:
  double lambda = DegreesToRadians(lat0); //to rad
  double phi = DegreesToRadians(lon0); //to rad
  double s = sin(lambda);
  double N_tmp = a_m / sqrt(1 - e_sq * s * s);

  double sin_lambda = sin(lambda);
  double cos_lambda = cos(lambda);
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);

  double x0 = (h0 + N_tmp) * cos_lambda * cos_phi;
  double y0 = (h0 + N_tmp) * cos_lambda * sin_phi;
  double z0 = (h0 + (1 - e_sq) * N_tmp) * sin_lambda;

  double xd = -sin_phi * E - cos_phi * sin_lambda * N + cos_lambda * cos_phi * U;
  double yd = cos_phi * E - sin_lambda * sin_phi * N + cos_lambda * sin_phi * U;
  double zd = cos_lambda * N + sin_lambda * U;

  *X = xd + x0;
  *Y = yd + y0;
  *Z = zd + z0;
}

void gnss::WGS84toENU(double lat, double lon, double h, double lat0, double lon0, double h0, double* E, double* N, double* U)
{
  double x, y, z;
  WGS84toECEF(lat, lon, h, &x, &y, &z);
  ECEFtoENU(x, y, z, lat0, lon0, h0, E, N, U);
}

void gnss::TWD97toWGS84(double E, double N, double *lat, double *lon, bool pkm)
{
  /*
  Convert coordintes from TWD97 to WGS84
  The east and north coordinates should be in meters and in float
  pkm true for Penghu, Kinmen and Matsu area
  You can specify one of the following presentations of the returned values:
      dms - A tuple with degrees (int), minutes (int) and seconds (float)
      dmsstr - [+/-]DDD°MMM'DDD.DDDDD" (unicode)
      mindec - A tuple with degrees (int) and minutes (float)
      mindecstr - [+/-]DDD°MMM.MMMMM' (unicode)
      (default)degdec - DDD.DDDDD (float)
  */

  double _lon0;
  if (pkm)
    _lon0 = lon0pkm;
  else
    _lon0 = lon0;

  E /= 1000.0;
  N /= 1000.0;

  double epsilon = (N-N0) / (k0*A);
  double eta = (E-E0) / (k0*A);

  double epsilonp = epsilon - beta1*sin(2*1*epsilon)*cosh(2*1*eta) - beta2*sin(2*2*epsilon)*cosh(2*2*eta) - beta3*sin(2*3*epsilon)*cosh(2*3*eta);
  double etap = eta - beta1*cos(2*1*epsilon)*sinh(2*1*eta) - beta2*cos(2*2*epsilon)*sinh(2*2*eta) - beta3*cos(2*3*epsilon)*sinh(2*3*eta);
  double sigmap = 1 - 2*1*beta1*cos(2*1*epsilon)*cosh(2*1*eta) - 2*2*beta2*cos(2*2*epsilon)*cosh(2*2*eta) - 2*3*beta3*cos(2*3*epsilon)*cosh(2*3*eta);
  double taup = 2*1*beta1*sin(2*1*epsilon)*sinh(2*1*eta) + 2*2*beta2*sin(2*2*epsilon)*sinh(2*2*eta) + 2*3*beta3*sin(2*3*epsilon)*sinh(2*3*eta);

  double chi = asin(sin(epsilonp) / cosh(etap));

  double latitude = chi + delta1*sin(2*1*chi) + delta2*sin(2*2*chi) + delta3*sin(2*3*chi);
  double longitude = _lon0 + atan(sinh(etap) / cos(epsilonp));
  *lat = latitude * DEG_PER_RAD;
  *lon = longitude * DEG_PER_RAD;
}

void gnss::WGS84toTWD97(double lat, double lon, double* E, double* N, bool pkm)
{
  /*
    Convert coordintes from WGS84 to TWD97
    pkm true for Penghu, Kinmen and Matsu area
    The latitude and longitude can be in the following formats:
        [+/-]DDD°MMM'SSS.SSSS" (unicode)
        [+/-]DDD°MMM.MMMM' (unicode)
        [+/-]DDD.DDDDD (string, unicode or float)
    The returned coordinates are in meters
  */

  double _lon0 = 0;
  if (pkm)
    _lon0 = lon0pkm;
  else
    _lon0 = lon0;

  lat = DegreesToRadians(lat);
  lon = DegreesToRadians(lon);

  double t = sinh((atanh(sin(lat)) - 2*pow(n,0.5)/(1+n)*atanh(2*pow(n,0.5)/(1+n)*sin(lat))));
  double epsilonp = atan(t/cos(lon-_lon0));
  double etap = atan(sin(lon-_lon0) / pow(1+t*t, 0.5));

  double E_tmp = E0 + k0*A*(etap + alpha1*cos(2*1*epsilonp)*sinh(2*1*etap) + alpha2*cos(2*2*epsilonp)*sinh(2*2*etap) + alpha3*cos(2*3*epsilonp)*sinh(2*3*etap));
  double N_tmp = N0 + k0*A*(epsilonp + alpha1*sin(2*1*epsilonp)*cosh(2*1*etap) + alpha2*sin(2*2*epsilonp)*cosh(2*2*etap) + alpha3*sin(2*3*epsilonp)*cosh(2*3*etap));

  *E = E_tmp*1000;
  *N = N_tmp*1000;
}

void gnss::RMCtoTWD97(char* RMC, double* E, double* N, double* Heading)
{
  /*
    Input : RMC from SBG Ekinox INS
      eg: $GPRMC,090110.25,A,2446.622573,N,12102.585780,E,0.049,16.78,211299,4.26,W,F,S*60
    Output : E (meter), N (meter), Heading (degree)
  */
    double latitude;
    double longitude;
    double fraction;
    char* pch;

    pch = strtok(RMC,","); //$GPRMC
    pch = strtok(NULL,","); //090110.25  hhmmss.ss
    pch = strtok(NULL,","); //A(Valid), V (NAV receiver warning)

    pch = strtok(NULL,","); //2446.622573  ddmm.mmmmmm
    latitude = atof(pch)/100;  //24.46622573
    fraction = modf (latitude, &latitude);  //fraction = 0.46622573, lat = 24
    latitude += fraction*100/60;  //24.777043
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //00212.181292   12102.602 dddmm.mmmmmm
    longitude = atof(pch)/100;  //2.12181292
    fraction = modf (longitude, &longitude);  //fraction = 0.12181292, lat = 2
    longitude += fraction*100/60; //121.043096
    // std::cout <<longitude <<"!"<<std::endl;
    pch = strtok(NULL,","); //E

    pch = strtok(NULL,","); //0.049 knot (1knot~1.852m/s) fff.f
    pch = strtok(NULL,","); //16.78 degree (heading direction, 0 degree = North, 90 degree = East) fff.f
    *Heading = atof(pch);
    pch = strtok(NULL,","); //211299 (1999/12/21) ddmmyy
    pch = strtok(NULL,","); //4.26     fff.ff
    pch = strtok(NULL,","); //W
    pch = strtok(NULL,","); //F
    pch = strtok(NULL,","); //S*60

    WGS84toTWD97(latitude,longitude,E,N);
}

void gnss::RMCtoTWD97(std::string RMC, double* E, double* N, double* Heading)
{
  /*
    Input : RMC from SBG Ekinox INS
      eg: $GPRMC,090110.25,A,2446.622573,N,12102.585780,E,0.049,16.78,211299,4.26,W,F,S*60
    Output : E (meter), N (meter), Heading (degree)
  */
    double latitude;
    double longitude;
    double fraction;
    char* pch;

    char *RMC_cstr = new char[RMC.length() + 1];
    strcpy(RMC_cstr, RMC.c_str());

    pch = strtok(RMC_cstr,","); //$GPRMC
    pch = strtok(NULL,","); //090110.25  hhmmss.ss
    pch = strtok(NULL,","); //A(Valid), V (NAV receiver warning)

    pch = strtok(NULL,","); //2446.622573  ddmm.mmmmmm
    latitude = atof(pch)/100;  //24.46622573
    fraction = modf (latitude, &latitude);  //fraction = 0.46622573, lat = 24
    latitude += fraction*100/60;  //24.777043
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //00212.181292   12102.602 dddmm.mmmmmm
    longitude = atof(pch)/100;  //2.12181292
    fraction = modf (longitude, &longitude);  //fraction = 0.12181292, lat = 2
    longitude += fraction*100/60; //121.043096
    // std::cout <<longitude <<"!"<<std::endl;
    pch = strtok(NULL,","); //E

    pch = strtok(NULL,","); //0.049 knot (1knot~1.852m/s) fff.f
    pch = strtok(NULL,","); //16.78 degree (heading direction, 0 degree = North, 90 degree = East) fff.f
    *Heading = atof(pch);
    pch = strtok(NULL,","); //211299 (1999/12/21) ddmmyy
    pch = strtok(NULL,","); //4.26     fff.ff
    pch = strtok(NULL,","); //W
    pch = strtok(NULL,","); //F
    pch = strtok(NULL,","); //S*60

    WGS84toTWD97(latitude,longitude,E,N);
}

void gnss::GGAtoTWD97(char* GGA, double* E, double* N)
{
  /*
    Input : GGA from SBG Ekinox INS
       $GPGGA,042326.00,2446.632644,N,12102.574917,E,4,15,0.0,113.724,M,19.581,M,15.0,0088*7C
    Output : E (meter), N (meter), Heading (degree)
  */

    double latitude;
    double longitude;
    double fraction;
    char* pch;

    pch = strtok(GGA,","); //$GPGGA
    pch = strtok(NULL,","); //042326.00  hhmmss.ss
    pch = strtok(NULL,","); //2446.632644  ddmm.mmmmmm

    latitude = atof(pch)/100;  //24.46632644
    fraction = modf (latitude, &latitude);  //fraction = 0.46632644, lat = 24
    latitude += fraction*100/60;  // 24.7772  (deg)
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //12102.574917    dddmm.mmmmmm
    longitude = atof(pch)/100;  //121.02574917
    fraction = modf (longitude, &longitude);  //fraction = 0.02574917, lat = 121
    longitude += fraction*100/60;  // 121.043  (deg)
    // std::cout <<longitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //E
    pch = strtok(NULL,","); //mode: 0 (autonomous), 1, 2, 3, 4(int), 5 (float RTK)
    pch = strtok(NULL,","); //satellite #
    pch = strtok(NULL,","); //HDOP
    pch = strtok(NULL,","); //Altitude 113.724
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Height of geoid above WGS84 ellipsoid 1.0
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Time since last DGPS update
    pch = strtok(NULL,","); //DGPS reference station id , checksum    0088*7C

    WGS84toTWD97(latitude,longitude,E,N);
}


void gnss::GGAtoTWD97(std::string GGA, double* E, double* N)
{
  /*
    Input : GGA from SBG Ekinox INS
       $GPGGA,042326.00,2446.632644,N,12102.574917,E,4,15,0.0,113.724,M,19.581,M,15.0,0088*7C
    Output : E (meter), N (meter), Heading (degree)
  */

    double latitude;
    double longitude;
    double fraction;
    char* pch;

    char *GGA_cstr = new char[GGA.length() + 1];
    strcpy(GGA_cstr, GGA.c_str());

    pch = strtok(GGA_cstr,","); //$GPGGA
    pch = strtok(NULL,","); //042326.00  hhmmss.ss
    pch = strtok(NULL,","); //2446.632644  ddmm.mmmmmm

    latitude = atof(pch)/100;  //24.46632644
    fraction = modf (latitude, &latitude);  //fraction = 0.46632644, lat = 24
    latitude += fraction*100/60;  // 24.7772  (deg)
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //12102.574917    dddmm.mmmmmm
    longitude = atof(pch)/100;  //121.02574917
    fraction = modf (longitude, &longitude);  //fraction = 0.02574917, lat = 121
    longitude += fraction*100/60;  // 121.043  (deg)
    // std::cout <<longitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //E
    pch = strtok(NULL,","); //mode: 0 (autonomous), 1, 2, 3, 4(int), 5 (float RTK)
    pch = strtok(NULL,","); //satellite #
    pch = strtok(NULL,","); //HDOP
    pch = strtok(NULL,","); //Altitude 113.724
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Height of geoid above WGS84 ellipsoid 1.0
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Time since last DGPS update
    pch = strtok(NULL,","); //DGPS reference station id , checksum    0088*7C

    WGS84toTWD97(latitude,longitude,E,N);
}


void gnss::GGAtoENU(char* GGA, double lat0, double lon0, double h0, double* E, double* N, double* U)
{
  /*
    Input : GGA from SBG Ekinox INS
       $GPGGA,042326.00,2446.632644,N,12102.574917,E,4,15,0.0,113.724,M,19.581,M,15.0,0088*7C
    Output : E (meter), N (meter), Heading (degree)
  */
    double latitude;
    double longitude;
    double altitude;
    double fraction;
    char* pch;

    pch = strtok(GGA,","); //$GPGGA
    pch = strtok(NULL,","); //042326.00  hhmmss.ss
    pch = strtok(NULL,","); //2446.632644  ddmm.mmmmmm

    latitude = atof(pch)/100;  //24.46632644
    fraction = modf (latitude, &latitude);  //fraction = 0.46632644, lat = 24
    latitude += fraction*100/60;  // 24.7772  (deg)
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //12102.574917    dddmm.mmmmmm
    longitude = atof(pch)/100;  //121.02574917
    fraction = modf (longitude, &longitude);  //fraction = 0.02574917, lat = 121
    longitude += fraction*100/60;  // 121.043  (deg)
    // std::cout <<longitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //E
    pch = strtok(NULL,","); //mode: 0 (autonomous), 1, 2, 3, 4(int), 5 (float RTK)
    pch = strtok(NULL,","); //satellite #
    pch = strtok(NULL,","); //HDOP
    pch = strtok(NULL,","); //Altitude 113.724
    altitude = atof(pch);
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Height of geoid above WGS84 ellipsoid 1.0
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Time since last DGPS update
    pch = strtok(NULL,","); //DGPS reference station id , checksum    0088*7C

    // std::cout << latitude << ", " << longitude << ", " << altitude << std::endl;

    WGS84toENU(latitude, longitude, altitude, lat0, lon0, h0, E, N, U);
}

void gnss::GGAtoENU(std::string GGA, double lat0, double lon0, double h0, double* E, double* N, double* U)
{
  /*
    Input : GGA from SBG Ekinox INS
       $GPGGA,042326.00,2446.632644,N,12102.574917,E,4,15,0.0,113.724,M,19.581,M,15.0,0088*7C
    Output : E (meter), N (meter), Heading (degree)
  */
    double latitude;
    double longitude;
    double altitude;
    double fraction;
    char* pch;

    char *GGA_cstr = new char[GGA.length() + 1];
    strcpy(GGA_cstr, GGA.c_str());

    pch = strtok(GGA_cstr,","); //$GPGGA
    pch = strtok(NULL,","); //042326.00  hhmmss.ss
    pch = strtok(NULL,","); //2446.632644  ddmm.mmmmmm

    latitude = atof(pch)/100;  //24.46632644
    fraction = modf (latitude, &latitude);  //fraction = 0.46632644, lat = 24
    latitude += fraction*100/60;  // 24.7772  (deg)
    // std::cout <<latitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //N
    pch = strtok(NULL,","); //12102.574917    dddmm.mmmmmm
    longitude = atof(pch)/100;  //121.02574917
    fraction = modf (longitude, &longitude);  //fraction = 0.02574917, lat = 121
    longitude += fraction*100/60;  // 121.043  (deg)
    // std::cout <<longitude <<"!"<<std::endl;

    pch = strtok(NULL,","); //E
    pch = strtok(NULL,","); //mode: 0 (autonomous), 1, 2, 3, 4(int), 5 (float RTK)
    pch = strtok(NULL,","); //satellite #
    pch = strtok(NULL,","); //HDOP
    pch = strtok(NULL,","); //Altitude 113.724
    altitude = atof(pch);
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Height of geoid above WGS84 ellipsoid 1.0
    pch = strtok(NULL,","); //M      (meter)
    pch = strtok(NULL,","); //Time since last DGPS update
    pch = strtok(NULL,","); //DGPS reference station id , checksum    0088*7C

    // std::cout << latitude << ", " << longitude << ", " << altitude << std::endl;

    WGS84toENU(latitude, longitude, altitude, lat0, lon0, h0, E, N, U);
}

double DegMinToDeg(double deg_min)
{
  double deg;
  double min;
  min = modf (deg_min, &deg);  
  return (deg + min/60);
}

double gnss::DegreesToRadians(double degrees)
{
    return degrees / DEG_PER_RAD;
}

double gnss::RadiansToDegrees(double radians)
{
    return radians * DEG_PER_RAD;
}

}