#ifndef GNSS_UTILITY_H_
#define GNSS_UTILITY_H_

namespace gnss_utility
{

class gnss
{
  public:
    gnss();

    void WGS84toECEF(double lat, double lon, double h, double* X, double* Y, double* Z);
    void ECEFtoWGS84(double X, double Y, double Z, double* lat, double* lon, double* h);
    void ECEFtoENU(double X, double Y, double Z, double lat0, double lon0, double h0, double* E, double* N, double* U);
    void ENUtoECEF(double E, double N, double U, double lat0, double lon0, double h0, double* X, double* Y, double* Z);
    void WGS84toENU(double lat, double lon, double h, double lat0, double lon0, double h0, double* E, double* N, double* U);

    void WGS84toTWD97(double lat, double lon, double* E, double* N, bool pkm=false);  //pkm true for Penghu, Kinmen and Matsu area, or false for Taiwan area
    void TWD97toWGS84(double E, double N, double* lat, double* lon, bool pkm=false);  //pkm true for Penghu, Kinmen and Matsu area, or false for Taiwan area
    void RMCtoTWD97(char* RMC, double* E, double* N, double* Heading);
    void RMCtoTWD97(std::string RMC, double* E, double* N, double* Heading);
    void GGAtoTWD97(char* GGA, double* E, double* N);
    void GGAtoTWD97(std::string GGA, double* E, double* N);
    
    void GGAtoENU(char* GGA, double lat0, double lon0, double h0, double* E, double* N, double* U);
    void GGAtoENU(std::string GGA, double lat0, double lon0, double h0, double* E, double* N, double* U);

    double DegMinToDeg(double deg_min);
    
    double DegreesToRadians(double degrees);
    double RadiansToDegrees(double radians);

  private:
    // used for
    const double PI; //3.14159265359
    const double DEG_PER_RAD; //(180.0/PI)
    const double a; // = 6378.137;    // WGS-84 Earth semimajor axis (km)
    const double b; // = 6356.752314245;  //Derived Earth semiminor axis (km)
    const double a_m; // WGS-84 Earth semimajor axis (m) : a*1000
    const double b_m; // WGS-84 Earth semimajor axis (m) : b*1000
    const double f; // = 1 / 298.257222101;  Ellipsoid Flatness : (a-b)/a
    const double k0; // = 0.9999;
    const double N0; // = 0;
    const double E0; // = 250.000;
    const double lon0; // = 121/DEG_PER_RAD; //radians(121)
    const double lon0pkm; // = 119/DEG_PER_RAD; //radians(119)

    const double n; // = f / (2-f);
    const double A; // = a / (1+n) * (1 + pow(n,2)/4.0 + pow(n,4)/64.0);
    const double alpha1; // = n/2 - 2*pow(n,2)/3.0 + 5*pow(n,3)/16.0;
    const double alpha2; // = 13*pow(n,2)/48.0 - 3*pow(n,3)/5.0;
    const double alpha3; // = 61*pow(n,3)/240.0;
    const double beta1; // = n/2 - 2*pow(n,2)/3.0 + 37*pow(n,3)/96.0;
    const double beta2; // = pow(n,2)/48.0 + pow(n,3)/15.0;
    const double beta3; // = 17*pow(n,3)/480.0;
    const double delta1; // = 2*n - 2*pow(n,2)/3.0 - 2*pow(n,3);
    const double delta2; // = 7*pow(n,2)/3.0 - 8*pow(n,3)/5.0;
    const double delta3; // = 56*pow(n,3)/15.0;


    const double e_sq;  // Square of Eccentricity :  f * (2 - f);

};

}

#endif  /* GNSS_UTILITY_H_ */
