/* Copyright(c) 2005 Trimble Navigation Ltd.
 * $Id: gsofParser.c,v 1.7 2011/11/15 20:37:11 tom Exp $
 * $Source: /home/CVS/panem/bin/src/gsofParser/gsofParser.c,v $
 *
 * gsofParser.c
 *
 * Parses stdin and extracts/lists GSOF information from it.
 *
 * The data is assumed to be the raw GSOF output from a Trimble
 * receiver.  That is, it consists of Trimcomm packets (02..03) of
 * type 0x40 in which are embedded GSOF subtype record192
 * This program accepts such data on standard input (either live as part
 * of a '|'-pipeline, or from a file via '<'-redirection.
 * It synchronizes with the individual Trimcomm packets and extracts
 * their contents.  When a complete set of GSOF-0x40 packets is
 * collected, the total contents is parsed and listed.  For some
 * GSOF subtypes there is a full decoder below and the contents will
 * be listed, item by item.  Other packets (for which I haven't bothered
 * to write a full decoder) will just be listed as Hex bytes.  Write
 * your own decoder if you need to, using the others as models.  It's
 * pretty simple.  Just tedious to implement every one of the subtypes.
 *
 * To understand this, start with main which collects Trimcomm packets.
 * Then move to postGsofData() which collects the GSOF data from
 * multiple packets and decides when a complete set has been received.
 * Then go to processGsofData() which steps through the collected data
 * parsing the individual gsof subtype records.  If the GSOF subtype is
 * one of the special ones where we have a decoder, that decoder is
 * called, otherwise we just dump the Hex bytes of the record.
 *
 * The program runs until the Stdinput indicates end of file [see gc()]
 * or the user kills it.
 *
 * NOTE: This program isn't designed to handle garbage data.  It may
 * choke on corrupted packets, etc.  Don't expect too much of it.
 * The idea is to be able to see the contents of well-formed GSOF
 * data, not to debug the overall formatting.  Though it should be
 * somewhat resistant to crazy data being mixed into the GSOF stream,
 * don't bet on it.
 */

#include <unistd.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>


//ros
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <tf/tf.h>

#define SERVER_PORT 8888
#define BUFF_LEN 1024

#define PI (3.14159265358979)

typedef unsigned long U32;
typedef unsigned short U16;
typedef signed short S16;
typedef char U8;

// A few global variables needed for collecting full GSOF packets from
// multiple Trimcomm packets.
char gsofData[2048];
int gsofDataIndex;

int ENU2LidXYZ_siwtch = 1;

double x_LiDAR, y_LiDAR, z_LiDAR, heading_LiDAR;
double base_lon,base_lat,base_h,T[3],R[3][3],T1[3],R1[3][3],T2[3],R2[3][3],T3[3],R3[3][3],T4[3],R4[3][3],T5[3],R5[3][3];
static sensor_msgs::Imu imu_data;
static sensor_msgs::Imu imu_data_rad;
static geometry_msgs::PoseStamped gnss_data;
static geometry_msgs::PoseStamped gnss2local_data;

static tf::Quaternion q_;
static ros::Publisher imu_pub;
static ros::Publisher imu_rad_pub;
static ros::Publisher gnss_pub;
static ros::Publisher gnss2local_pub;


int close(int fd);

/**********************************************************************/
void initial_para()
/**********************************************************************/
{
        double read_tmp[63];
        int read_index = 0;
        std::string fname = ros::package::getPath("trimble_gps_imu_pub");
        fname += "/data/ITRI_NEW_ENU2LidXYZ_sec.txt";
        std::cout << fname << std::endl;

        std::ifstream fin;
        char line[100];
        memset( line, 0, sizeof(line));

        fin.open(fname.c_str(),std::ios::in);
        if(!fin)
        {
                std::cout << "Fail to import txt" <<std::endl;
                exit(1);
        }
        while(fin.getline(line,sizeof(line),','))
        {
                // fin.getline(line,sizeof(line),'\n');
                std::string nmea_str(line);
                std::stringstream ss(nmea_str);
                std::string token;

                getline(ss,token, ',');
                read_tmp[read_index] = atof(token.c_str());
                read_index += 1;
        }

        std::cout << read_tmp[10] << std::endl;
        base_lon = read_tmp[0];
        base_lat = read_tmp[1];
        base_h = read_tmp[2];
        if (ENU2LidXYZ_siwtch == 0)
        {
        	R[0][0] = read_tmp[3];
	        R[0][1] = read_tmp[4];
	        R[0][2] = read_tmp[5];
	        R[1][0] = read_tmp[6];
	        R[1][1] = read_tmp[7];
	        R[1][2] = read_tmp[8];
	        R[2][0] = read_tmp[9];
	        R[2][1] = read_tmp[10];
	        R[2][2] = read_tmp[11];
	        T[0] = read_tmp[12];
	        T[1] = read_tmp[13];
	        T[2] = read_tmp[14];
        }
        else
        {
        	R1[0][0] = read_tmp[3];
	        R1[0][1] = read_tmp[4];
	        R1[0][2] = read_tmp[5];
	        R1[1][0] = read_tmp[6];
	        R1[1][1] = read_tmp[7];
	        R1[1][2] = read_tmp[8];
	        R1[2][0] = read_tmp[9];
	        R1[2][1] = read_tmp[10];
	        R1[2][2] = read_tmp[11];
	        R2[0][0] = read_tmp[12];
	        R2[0][1] = read_tmp[13];
	        R2[0][2] = read_tmp[14];
	        R2[1][0] = read_tmp[15];
	        R2[1][1] = read_tmp[16];
	        R2[1][2] = read_tmp[17];
	        R2[2][0] = read_tmp[18];
	        R2[2][1] = read_tmp[19];
	        R2[2][2] = read_tmp[20];
	        R3[0][0] = read_tmp[21];
	        R3[0][1] = read_tmp[22];
	        R3[0][2] = read_tmp[23];
	        R3[1][0] = read_tmp[24];
	        R3[1][1] = read_tmp[25];
	        R3[1][2] = read_tmp[26];
	        R3[2][0] = read_tmp[27];
	        R3[2][1] = read_tmp[28];
	        R3[2][2] = read_tmp[29];
	        R4[0][0] = read_tmp[30];
	        R4[0][1] = read_tmp[31];
	        R4[0][2] = read_tmp[32];
	        R4[1][0] = read_tmp[33];
	        R4[1][1] = read_tmp[34];
	        R4[1][2] = read_tmp[35];
	        R4[2][0] = read_tmp[36];
	        R4[2][1] = read_tmp[37];
	        R4[2][2] = read_tmp[38];
	        R5[0][0] = read_tmp[39];
	        R5[0][1] = read_tmp[40];
	        R5[0][2] = read_tmp[41];
	        R5[1][0] = read_tmp[42];
	        R5[1][1] = read_tmp[43];
	        R5[1][2] = read_tmp[44];
	        R5[2][0] = read_tmp[45];
	        R5[2][1] = read_tmp[46];
	        R5[2][2] = read_tmp[47];
	        T1[0] = read_tmp[48];
	        T1[1] = read_tmp[49];
	        T1[2] = read_tmp[50];
	        T2[0] = read_tmp[51];
	        T2[1] = read_tmp[52];
	        T2[2] = read_tmp[53];
	        T3[0] = read_tmp[54];
	        T3[1] = read_tmp[55];
	        T3[2] = read_tmp[56];
	        T4[0] = read_tmp[57];
	        T4[1] = read_tmp[58];
	        T4[2] = read_tmp[59];
	        T5[0] = read_tmp[60];
	        T5[1] = read_tmp[61];
	        T5[2] = read_tmp[62];
	        std::cout << "T[5] : " << std::setprecision(20) << T5[0] << std::endl;
        }
        std::cout << "init_long : " << std::setprecision(20) << base_lon << std::endl;
        std::cout << "init_lat : " << std::setprecision(20) << base_lat << std::endl;
        std::cout << "init_alt : " << std::setprecision(20) << base_h << std::endl;
}

/**********************************************************************/
void GPStoLiDARCoordinate(double &lat, double &lon, double &alt)
/**********************************************************************/
{
        long a = 6378137;
        double b = 6356752.3142;
        double e2 = 1 - (b / a) * (b / a);
        double phi, lam, h, dphi, dlam, dh, tmp1, cp, sp, de, dn, du;
        double R_final[3][3],T_final[3];
        // double base_lat = 22.9232719; // 自駕車 精準地圖 緯度
        // double base_lon = 120.2882873; // 自駕車 精準地圖 經度
        // double base_h = 26.688; //

        phi = base_lat * PI / 180;
        lam = base_lon * PI / 180;
        h = base_h;

        dphi = lat * PI / 180 - phi; // rv_latitude:自車緯度
        dlam = lon * PI / 180 - lam; // rv_longitude:自車經度
        dh = alt - h;

        tmp1 = sqrt(1 - e2 * sin(phi) * sin(phi));

        cp = cos(phi);
        sp = sin(phi);

        de = (a / tmp1 + h) * cp * dlam - (a * (1 - e2) / (tmp1 * tmp1 * tmp1) + h) * sp * dphi * dlam + cp * dlam * dh;
        dn = (a * (1 - e2) / (tmp1 * tmp1 * tmp1) + h) * dphi + 1.5 * cp * sp * a * e2 * dphi * dphi
             + 0.5 * sp * cp * (a / tmp1 + h) * dlam * dlam + sp * sp * dh * dphi;
        du = dh -0.5 * (a - 1.5 * a * e2 * cp * cp + 0.5 * a * e2 + h) * dphi * dphi
             - 0.5 * cp * cp * (a / tmp1 - h) * dlam * dlam;
        double d[3] = { de, dn, du };
        
        if (ENU2LidXYZ_siwtch == 0)
        {
        	for (int i = 0;i < 3;i++)
	        {
	        	for (int j = 0; j < 3;j++)
	        	{
	        		R_final[i][j] = R[i][j];
	        		T_final[i] = T[i];
	        	}
	        }
        }
        else
        {
        	for (int i = 0;i < 3;i++)
	        {
	        	for (int j = 0; j < 3;j++)
	        	{
	        		if (de <= 0)
	        		{
	        			R_final[i][j] = R1[i][j];
	        			T_final[i] = T1[i];
	        		}
	        		else if (de > 0 && de <= 100)
	        		{
	        			R_final[i][j] = R2[i][j];
	        			T_final[i] = T2[i];
	        		}
	        		else if (de > 100 && de <= 225)
	        		{
	        			R_final[i][j] = R3[i][j];
	        			T_final[i] = T3[i];
	        		}
	        		else if (de > 225 && de <= 350)
	        		{
	        			R_final[i][j] = R4[i][j];
	        			T_final[i] = T4[i];
	        		}
	        		else
	        		{
	        			R_final[i][j] = R5[i][j];
	        			T_final[i] = T5[i];
	        		}
	        	}
	        	
	        }
        }

        // double R[3][3] = { { 0.988929666694072, -0.025626152042906, 0 },
        //                    { 0.025626152042906, 0.988929666694072, 0 },
        //                    { 0,                 0,                 0.989261636442137 } };
        // double T[3] = { -98.681792834410710, 146.1071686888813, 0 };
        double d_new[3];
        for (int i = 0; i < sizeof(d)/sizeof(d[0]); i++) {
                d_new[i] = R_final[i][0] * d[0] + R_final[i][1] * d[1] + R_final[i][2] * d[2] + T_final[i];
        }

        printf("  de: %f, dn: %f, du: %f\n",de,dn,du);
        printf("  de_LiD: %f, dn_LiD: %f, du_LiD: %f\n", d_new[0],d_new[1],d_new[2]);



        lat = d_new[0];
        lon = d_new[1];
        alt = d_new[2];


}

/**********************************************************************/
unsigned long getU32( char * * ppData )
/**********************************************************************/
// Used by the decoding routines to grab 4 bytes and pack them into
// a U32.  Fed ppData which is a pointer to a pointer to the start of
// the data bytes.  The pointer variable referenced by ppData is moved
// beyond the four bytes.
// This is designed to work on little-endian processors (Like Pentiums).
// Effectively that means we reverse the order of the bytes.
// This would need to be rewritten to work on big-endian PowerPCs.
{
        unsigned long retValue;
        char * pBytes;

        pBytes = (char *)(&retValue) + 3;

        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes   = *(*ppData)++;

        return retValue;

} /* end of getU32() */



/**********************************************************************/
float getFloat( char * * ppData )
/**********************************************************************/
// Used by the decoding routines to grab 4 bytes and pack them into
// a Float.  Fed ppData which is a pointer to a pointer to the start of
// the data bytes.  The pointer variable referenced by ppData is moved
// beyond the four bytes.
// This is designed to work on little-endian processors (Like Pentiums).
// Effectively that means we reverse the order of the bytes.
// This would need to be rewritten to work on big-endian PowerPCs.
{
        float retValue;
        char * pBytes;

        pBytes = (char *)(&retValue) + 3;


        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes   = *(*ppData)++;

        return retValue;

} /* end of getFloat() */



/**********************************************************************/
double getDouble( char * * ppData )
/**********************************************************************/
// Used by the decoding routines to grab 8 bytes and pack them into
// a Double.  Fed ppData which is a pointer to a pointer to the start of
// the data bytes.  The pointer variable referenced by ppData is moved
// beyond the four bytes.
// This is designed to work on little-endian processors (Like Pentiums).
// Effectively that means we reverse the order of the bytes.
// This would need to be rewritten to work on big-endian PowerPCs.
{
        double retValue;
        char * pBytes;

        pBytes = (char *)(&retValue) + 7;


        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes-- = *(*ppData)++;
        *pBytes   = *(*ppData)++;

        return retValue;

} /* end of getDouble() */



/**********************************************************************/
unsigned short getU16( char * * ppData )
/**********************************************************************/
// Used by the decoding routines to grab 2 bytes and pack them into
// a U16.  Fed ppData which is a pointer to a pointer to the start of
// the data bytes.  The pointer variable referenced by ppData is moved
// beyond the four bytes.
// This is designed to work on little-endian processors (Like Pentiums).
// Effectively that means we reverse the order of the bytes.
// This would need to be rewritten to work on big-endian PowerPCs.
{
        unsigned short retValue;
        char * pBytes;

        pBytes = (char *)(&retValue) + 1;

        *pBytes-- = *(*ppData)++;
        *pBytes   = *(*ppData)++;

        return retValue;

} /* end of getU16() */




/***********************************************************************
 * The next section contains routines which are parsers for individual
 * GSOF records.  They are all passed a length (which is listed but
 * usually not used) and a pointer to the data bytes that make up the
 * record.
 ***********************************************************************
 */


/**********************************************************************/
void processPositionTime( int length, char *pData )
/**********************************************************************/
{
        unsigned long msecs;
        unsigned short weekNumber;
        int nSVs;
        int flags1;
        int flags2;
        int initNumber;

        printf( "  GsofType:1 - PositionTime  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        msecs = getU32( &pData );
        weekNumber = getU16( &pData );
        nSVs = *pData++;
        flags1 = *pData++;
        flags2 = *pData++;
        initNumber = *pData++;

        printf( "  Milliseconds:%ld  Week:%d  #Svs:%d "
                "flags:%02X:%02X init:%d\n",
                msecs,
                weekNumber,
                nSVs,
                flags1,
                flags2,
                initNumber
                );


} /* end of processPositionTime() */





/**********************************************************************/
void processLatLonHeight( int length, char *pData )
/**********************************************************************/
{
        double lat, lon, height;

        printf( "  GsofType:2 - LatLongHeight   len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif
        lat = getDouble( &pData ) * 180.0 / PI;
        lon = getDouble( &pData ) * 180.0 / PI;
        height = getDouble( &pData );

        printf( "  Lat:%.7f Lon:%.7f Height:%.3f\n",
                lat,
                lon,
                height
                );
} /* end of processLatLonHeight() */





/**********************************************************************/
void processECEF( int length, char *pData )
/**********************************************************************/
{
        double X, Y, Z;

        printf( "  GsofType:3 - ECEF   len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif
        X = getDouble( &pData );
        Y = getDouble( &pData );
        Z = getDouble( &pData );

        printf( "  X:%.3f Y:%.3f Z:%.3f\n", X, Y, Z );

} /* end of processECEF() */



/**********************************************************************/
void processLocalDatum( int length, char *pData )
/**********************************************************************/
{
        char id[9];
        double lat, lon, height;

        printf( "  GsofType:4 - Local Datum Position  "
                "!!!!!UNTESTED!!!!!!!  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        memcpy( id, pData, 8 );
        pData += 8;
        // id[9] = 0;  // Out of bound

        lat = getDouble( &pData ) * 180.0 / PI;
        lon = getDouble( &pData ) * 180.0 / PI;
        height = getDouble( &pData );

        printf( "  Id:%s Lat:%.7f Lon:%.7f Height:%.3f\n",
                id,
                lat,
                lon,
                height
                );
} /* end of processLocalDatum() */



/**********************************************************************/
void processEcefDelta( int length, char *pData )
/**********************************************************************/
{
        double X, Y, Z;

        printf( "  GsofType:6 - ECEF Delta  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        X = getDouble( &pData );
        Y = getDouble( &pData );
        Z = getDouble( &pData );

        printf( "  X:%.3f Y:%.3f Z:%.3f\n", X, Y, Z );

} /* end of processEcefDelta() */



/**********************************************************************/
void processTangentPlaneDelta( int length, char *pData )
/**********************************************************************/
{
        double E, N, U;

        printf( "  GsofType:7 - Tangent Plane Delta  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        E = getDouble( &pData );
        N = getDouble( &pData );
        U = getDouble( &pData );

        printf( "  East:%.3f North:%.3f Up:%.3f\n", E, N, U );

} /* end of processTangentPlaneDelta() */



/**********************************************************************/
void processVelocityData( int length, char *pData )
/**********************************************************************/
{
        int flags;
        float velocity;
        float heading;
        float vertical;

        printf( "  GsofType:8 - Velocity Data  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        flags = *pData++;

        velocity = getFloat( &pData );
        heading = getFloat( &pData ) * 180.0 / PI;
        vertical = getFloat( &pData );

        printf( "  Flags:%02X  velocity:%.3f  heading:%.3f  vertical:%.3f\n",
                flags,
                velocity,
                heading,
                vertical
                );

} /* end of processVelocityData() */



/**********************************************************************/
void processUtcTime( int length, char *pData )
/**********************************************************************/
{

        printf( "  GsofType:16 - UTC Time Info   len:%d\n",
                length
                );

        U32 msecs = getU32( &pData );
        U16 weekNumber = getU16( &pData );
        S16 utcOffset = getU16( &pData );
        U8 flags = *pData++;

        printf( "  ms:%lu  week:%u  utcOff:%d  flags:%02x\n",
                msecs,
                weekNumber,
                utcOffset,
                flags
                );

} /* end of processUtcTime() */



/**********************************************************************/
void processPdopInfo( int length, char *pData )
/**********************************************************************/
{
        float pdop;
        float hdop;
        float vdop;
        float tdop;

        printf( "  GsofType:9 - PDOP Info   len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        pdop = getFloat( &pData );
        hdop = getFloat( &pData );
        vdop = getFloat( &pData );
        tdop = getFloat( &pData );

        printf( "  PDOP:%.1f  HDOP:%.1f  VDOP:%.1f  TDOP:%.1f\n",
                pdop,
                hdop,
                vdop,
                tdop
                );

} /* end of processPdopInfo() */



/**********************************************************************/
void processBriefSVInfo( int length, char *pData )
/**********************************************************************/
{
        int nSVs;
        int i;

        printf( "  GsofType:13 - SV Brief Info   len:%d\n",
                length
                );

        nSVs = *pData++;
        printf( "  SvCount:%d\n", nSVs );

        for ( i = 0; i < nSVs; ++i )
        {
                int prn;
                int flags1;
                int flags2;

                prn = *pData++;
                flags1 = *pData++;
                flags2 = *pData++;

                printf( "  Prn:%-2d  flags:%02X:%02X\n", prn, flags1, flags2 );
        }
} /* end of processBriefSVInfo */



/**********************************************************************/
void processAllBriefSVInfo( int length, char *pData )
/**********************************************************************/
{
        int nSVs;
        int i;

        printf( "  GsofType:33 - All SV Brief Info   len:%d\n",
                length
                );

        nSVs = *pData++;
        printf( "  SvCount:%d\n", nSVs );

        for ( i = 0; i < nSVs; ++i )
        {
                int prn;
                int system;
                int flags1;
                int flags2;

                prn = *pData++;
                system = *pData++;
                flags1 = *pData++;
                flags2 = *pData++;

                printf( "  %s SV:%-2d  flags:%02X:%02X\n",
                        system == 0 ? "GPS"
                        : system == 1 ? "SBAS"
                        : system == 2 ? "GLONASS"
                        : system == 3 ? "GALILEO" : "RESERVED",
                        prn, flags1, flags2 );
        }
} /* end of processAllBriefSVInfo */



/**********************************************************************/
void processAllDetailedSVInfo( int length, char *pData )
/**********************************************************************/
{
        int nSVs;
        int i;

        printf( "  GsofType:34 - All SV Detailed Info   len:%d\n",
                length
                );

        nSVs = *pData++;
        printf( "  SvCount:%d\n", nSVs );

        for ( i = 0; i < nSVs; ++i )
        {
                int prn;
                int system;
                int flags1;
                int flags2;
                int elevation;
                int azimuth;
                int snr[ 3 ];

                prn = *pData++;
                system = *pData++;
                flags1 = *pData++;
                flags2 = *pData++;
                elevation = *pData++;
                azimuth = getU16( &pData );
                snr[ 0 ] = *pData++;
                snr[ 1 ] = *pData++;
                snr[ 2 ] = *pData++;

                printf( "  %s SV:%-2d  flags:%02X:%02X\n"
                        "     El:%2d  Az:%3d\n"
                        "     SNR %3s %5.2f\n"
                        "     SNR %3s %5.2f\n"
                        "     SNR %3s %5.2f\n",
                        system == 0 ? "GPS"
                        : system == 1 ? "SBAS"
                        : system == 2 ? "GLONASS"
                        : system == 3 ? "GALILEO" : "RESERVED",
                        prn, flags1, flags2,
                        elevation, azimuth,
                        system == 3 ? "E1 " : "L1 ", (float)snr[ 0 ] / 4.0,
                        system == 3 ? "N/A " : "L2 ", (float)snr[ 1 ] / 4.0,
                        system == 3 ? "E5 "
                        : system == 2 ? "G1P" : "L5 ", (float)snr[ 2 ] / 4.0
                        );
        }
} /* end of processAllDetailedSVInfo */



/**********************************************************************/
void processSvDetailedInfo( int length, char *pData )
/**********************************************************************/
{
        int nSVs;
        int i;

        printf( "  GsofType:14 - SV Detailed Info   len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        nSVs = *pData++;
        printf( "  SvCount:%d\n", nSVs );

        for ( i = 0; i < nSVs; ++i )
        {
                int prn;
                int flags1;
                int flags2;
                int elevation;
                int azimuth;
                int l1Snr;
                int l2Snr;

                prn = *pData++;
                flags1 = *pData++;
                flags2 = *pData++;
                elevation = *pData++;
                azimuth = getU16( &pData );
                l1Snr = *pData++;
                l2Snr = *pData++;

                printf( "   Prn:%-2d  flags:%02X:%02X elv:%-2d azm:%-3d  "
                        "L1snr:%-5.2f L2snr:%-5.2f\n",
                        prn,
                        flags1,
                        flags2,
                        elevation,
                        azimuth,
                        ((double)l1Snr) / 4.0,
                        ((double)l2Snr) / 4.0
                        );
        }
} /* end of processSvDetailedInfo() */



/**********************************************************************/
void processPositionTimeUtc( int length, char *pData )
/**********************************************************************/
{
        unsigned long msecs;
        unsigned short weekNumber;
        signed short utcOffset;

        printf( "  GsofType:26 - PositionTimeUtc  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        msecs = getU32( &pData );
        weekNumber = getU16( &pData );
        utcOffset = (signed short)getU16( &pData );

        printf( "  Milliseconds:%ld  Week:%d UTCoffset:%d\n",
                msecs,
                weekNumber,
                utcOffset
                );

} /* end of processPositionTimeUtc() */



/**********************************************************************/
void processAttitudeInfo( int length, char *pData )
/**********************************************************************/
{
        double gpsTime;
        char flags;
        char nSVs;
        char mode;
        double pitch;
        double yaw;
        double roll;
        double range;
        double pdop;

        printf( "  GsofType:27 - AttitudeInfo  len:%d\n",
                length
                );

  #if 0
        {
                int i;
                for ( i = 0; i < length; ++i )
                {
                        printf( "%02X%c",
                                pData[i],
                                i % 16 == 15 ? '\n' : ' '
                                );
                }
                printf( "\n" );
        }
  #endif

        gpsTime = (double)getU32( &pData ) / 1000.0;
        flags = *pData++;
        nSVs = *pData++;
        mode = *pData++;
        ++pData; // reserved
        pitch = getDouble( &pData ) / PI * 180.0;
        yaw   = getDouble( &pData ) / PI * 180.0;
        roll  = getDouble( &pData ) / PI * 180.0;
        range = getDouble( &pData );

        pdop  = (double)getU16( &pData ) / 10.0;

        printf( "  Time:%.3f"
                " flags:%02X"
                " nSVs:%d"
                " mode:%d\n"
                "  pitch:%.3f"
                " yaw:%.3f"
                " roll:%.3f"
                " range:%.3f"
                " pdop:%.1f"
                "\n",
                gpsTime,
                flags,
                nSVs,
                mode,
                pitch,
                yaw,
                roll,
                range,
                pdop
                );

        // Detect if the extended record information is present
        if ( length > 42 )
        {
                float pitch_var;
                float yaw_var;
                float roll_var;
                float pitch_yaw_covar;
                float pitch_roll_covar;
                float yaw_roll_covar;
                float range_var;

                // The variances are in units of radians^2
                pitch_var = getFloat( &pData );
                yaw_var   = getFloat( &pData );
                roll_var  = getFloat( &pData );

                // The covariances are in units of radians^2
                pitch_yaw_covar  = getFloat( &pData );
                pitch_roll_covar = getFloat( &pData );
                yaw_roll_covar   = getFloat( &pData );

                // The range variance is in units of m^2
                range_var = getFloat( &pData );

                printf( "  variance (radians^2)"
                        " pitch:%.4e"
                        " yaw:%.4e"
                        " roll:%.4e"
                        "\n",
                        pitch_var,
                        yaw_var,
                        roll_var );

                printf( "  covariance (radians^2)"
                        " pitch-yaw:%.4e"
                        " pitch-roll:%.4e"
                        " yaw-roll:%.4e"
                        "\n",
                        pitch_yaw_covar,
                        pitch_roll_covar,
                        yaw_roll_covar );

                printf( "  variance (m^2)"
                        " range: %.4e"
                        "\n",
                        range_var );
        }

} /* end of processAttitudeInfo() */


/**********************************************************************/
void processLbandStatus( int length, char *pData )
/**********************************************************************/
{
        char name[5];
        float freq;
        unsigned short bit_rate;
        float snr;
        char hp_xp_subscribed_engine;
        char hp_xp_library_mode;
        char vbs_library_mode;
        char beam_mode;
        char omnistar_motion;
        float horiz_prec_thresh;
        float vert_prec_thresh;
        char nmea_encryption;
        float iq_ratio;
        float est_ber;
        unsigned long total_uw;
        unsigned long total_bad_uw;
        unsigned long total_bad_uw_bits;
        unsigned long total_viterbi;
        unsigned long total_bad_viterbi;
        unsigned long total_bad_messages;
        char meas_freq_is_valid = -1;
        double meas_freq = 0.0;

        printf( "  GsofType:40 - LBAND status  len:%d\n",
                length
                );

        memcpy( name, pData, 5 );
        pData += 5;
        freq = getFloat( &pData );
        bit_rate = getU16( &pData );
        snr = getFloat( &pData );
        hp_xp_subscribed_engine = *pData++;
        hp_xp_library_mode = *pData++;
        vbs_library_mode = *pData++;
        beam_mode = *pData++;
        omnistar_motion = *pData++;
        horiz_prec_thresh = getFloat( &pData );
        vert_prec_thresh = getFloat( &pData );
        nmea_encryption = *pData++;
        iq_ratio = getFloat( &pData );
        est_ber = getFloat( &pData );
        total_uw = getU32( &pData );
        total_bad_uw = getU32( &pData );
        total_bad_uw_bits = getU32( &pData );
        total_viterbi = getU32( &pData );
        total_bad_viterbi = getU32( &pData );
        total_bad_messages = getU32( &pData );
        if( length > 61 )
        {
                meas_freq_is_valid = *pData++;
                meas_freq = getDouble( &pData );
        }

        printf( "  Name:%s"
                "  Freq:%g"
                "  bit rate:%d"
                "  SNR:%g"
                "\n"
                "  HP/XP engine:%d"
                "  HP/XP mode:%d"
                "  VBS mode:%d"
                "\n"
                "  Beam mode:%d"
                "  Omnistar Motion:%d"
                "\n"
                "  Horiz prec. thresh.:%g"
                "  Vert prec. thresh.:%g"
                "\n"
                "  NMEA encryp.:%d"
                "  I/Q ratio:%g"
                "  Estimated BER:%g"
                "\n"
                "  Total unique words(UW):%lu"
                "  Bad UW:%lu"
                "  Bad UW bits:%lu"
                "\n"
                "  Total Viterbi:%lu"
                "  Corrected Viterbi:%lu"
                "  Bad messages:%lu"
                "\n"
                "  Meas freq valid?:%d"
                "  Meas freq:%.3f"
                "\n"
                ,
                name,
                freq,
                bit_rate,
                snr,
                hp_xp_subscribed_engine,
                hp_xp_library_mode,
                vbs_library_mode,
                beam_mode,
                omnistar_motion,
                horiz_prec_thresh,
                vert_prec_thresh,
                nmea_encryption,
                iq_ratio,
                est_ber,
                total_uw,
                total_bad_uw,
                total_bad_uw_bits,
                total_viterbi,
                total_bad_viterbi,
                total_bad_messages,
                meas_freq_is_valid,
                meas_freq
                );

} /* end of processLbandStatus() */

/**********************************************************************/
void processINSFullNavigation( int length, char *pData )
/**********************************************************************/
{
        unsigned short GPS_week_number;
        unsigned int GPS_time_ms;
        char IMU_alignment_status;
        char GPS_quality_indicator;
        double Latitude;
        double Longitude;
        double Altitude;
        float North_Velocity;
        float East_Velocity;
        float Down_velocity;
        float Total_Speed;
        double Roll;
        double Pitch;
        double Heading;
        double Track_Angle;
        float Angular_rate_Long_X;
        float Angular_rate_Traverse_Y;
        float Angular_rate_Down_Z;
        float Longitudinal_accel_X;
        float Traverse_accel_Y;
        float Down_accel_Z;

        double Latitude_local;
        double Longitude_local;
        double Altitude_local;

        /****************************************************************************/
        /*                                                                          */
        /*    IMU Alignment Status provides the status of the BX/BD935 INS Solution */
        /*                                                                          */
        /*    0 - GPS Only                                                          */
        /*    1 - Coarse leveling                                                   */
        /*    2 - Degraded solution                                                 */
        /*    3 - Aligned                                                           */
        /*    4 - Full navigation mode                                              */
        /*                                                                          */
        /****************************************************************************/

        /****************************************************************************/
        /*                                                                          */
        /*    GPS quality indicator maps the internal rover fix to GNSS quality     */
        /*    (same as in the NMEA GGA Protocol Message).                           */
        /*                                                                          */
        /*  0 - Fix not available or invalid                                      */
        /*  1 - GPS SPS Mode, fix valid                                           */
        /*  2 - Differential GPS, SPS Mode, fix valid                             */
        /*  3 - GPS PPS Mode, fix valid                                           */
        /*  4 - Real Time Kinematic. System used in RTK mode with fixed integers  */
        /*  5 - Float RTK. Satellite system used in RTK mode, floating integers   */
        /*  6 - Estimated (dead reckoning) Mode                                   */
        /*  7 - Manual Input Mode                                                 */
        /*  8 - Simulator Mode                                                    */
        /*                                                                          */
        /****************************************************************************/


        printf( "  -------------- GsofType: 49 - INS Full Navigation Info - Length (in Bytes):%d ------------ \n", length);
        printf( " \n " );

        GPS_week_number = getU16( &pData );

        GPS_time_ms =  (int) getU32( &pData );

        IMU_alignment_status = *pData++;
        GPS_quality_indicator = *pData++;

        Latitude = getDouble( &pData );
        Longitude = getDouble( &pData );
        Altitude = getDouble( &pData );

        Latitude_local = Latitude;
        Longitude_local = Longitude;
        Altitude_local = Altitude;

        North_Velocity = getFloat( &pData );
        East_Velocity = getFloat( &pData );
        Down_velocity = getFloat( &pData );
        Total_Speed = getFloat( &pData );

        Roll = getDouble( &pData );
        Pitch = getDouble( &pData );
        Heading = getDouble( &pData );
        Track_Angle = getDouble( &pData );
	Roll = Roll*M_PI / 180;
	Pitch = Pitch*M_PI / 180;
	Heading = Heading*M_PI / 180;
        Angular_rate_Long_X = getFloat( &pData );
        Angular_rate_Traverse_Y = getFloat( &pData );
        Angular_rate_Down_Z = getFloat( &pData );

        Longitudinal_accel_X = getFloat( &pData );
        Traverse_accel_Y = getFloat( &pData );
        Down_accel_Z = getFloat( &pData );

        printf( "  GPS_Week:%d TOWms:%u IMU_Status:%02X \n", GPS_week_number, GPS_time_ms, IMU_alignment_status );
        printf( " \n " );
        printf( "  GPS Quality:%02X Lat:%.7lf Long:%.7lf Alt:%.3lf \n", GPS_quality_indicator,  Latitude, Longitude, Altitude );
        printf( " \n " );
        printf( "  N Vel:%.3lf E Vel:%.3lf D Vel:%.3lf Total Sp:%.3lf \n", North_Velocity, East_Velocity, Down_velocity, Total_Speed );
        printf( " \n " );
        printf( "  Roll:%.3lf Pitch:%.3lf Heading:%.3lf Track Angle:%.3lf  \n", Roll, Pitch, Heading, Track_Angle );
        printf( " \n " );
        printf( "  Angular Rates (Deg/sec) - Long_X:%.3lf Traverse_Y:%.3lf Down_Z:%.3lf \n", Angular_rate_Long_X, Angular_rate_Traverse_Y, Angular_rate_Down_Z );
        printf( " \n " );
        printf( "  Accelerations (m/s^2) -   Long_X:%.3lf Traverse_Y:%.3lf Down_Z:%.3lf \n", Longitudinal_accel_X, Traverse_accel_Y, Down_accel_Z );
        printf( " \n " );
        printf( " -------------------------------------End INS Type 49------------------------- \n ");

        tf::Quaternion gnss_q;

        gnss_data.header.stamp = ros::Time::now();
        gnss_data.header.frame_id = "map";
        //linear
        gnss_data.pose.position.x = Latitude;
        gnss_data.pose.position.y = Longitude;
        gnss_data.pose.position.z = Altitude;
        gnss_q.setRPY(Roll, Pitch, Heading);

        gnss_data.pose.orientation.x = gnss_q.x();
        gnss_data.pose.orientation.y = gnss_q.y();
        gnss_data.pose.orientation.z = gnss_q.z();
        gnss_data.pose.orientation.w = gnss_q.w();
        gnss_pub.publish(gnss_data);

        GPStoLiDARCoordinate(Latitude_local, Longitude_local, Altitude_local);

        tf::Quaternion gnss2local_q;

        gnss2local_data.header.stamp = ros::Time::now();
        gnss2local_data.header.frame_id = "map";

        gnss2local_data.pose.position.x = Latitude_local;
        gnss2local_data.pose.position.y = Longitude_local;
        gnss2local_data.pose.position.z = Altitude_local;
        gnss2local_q.setRPY(Roll, Pitch, Heading);

        gnss2local_data.pose.orientation.x = gnss2local_q.x();
        gnss2local_data.pose.orientation.y = gnss2local_q.y();
        gnss2local_data.pose.orientation.z = gnss2local_q.z();
        gnss2local_data.pose.orientation.w = gnss2local_q.w();
        gnss2local_pub.publish(gnss2local_data);


        q_.setRPY(Roll, Pitch, Heading);

        // raw data
        imu_data.header.stamp = ros::Time::now();
        imu_data.header.frame_id = "map";
        //Qua
        imu_data.orientation.x = q_.x();
        imu_data.orientation.y = q_.y();
        imu_data.orientation.z = q_.z();
        imu_data.orientation.w = q_.w();

        //linear_acceleration
        imu_data.linear_acceleration.x = Longitudinal_accel_X;
        imu_data.linear_acceleration.y = Traverse_accel_Y;
        imu_data.linear_acceleration.z = Down_accel_Z;

        //angular_velocity
        imu_data.angular_velocity.x = Angular_rate_Long_X;
        imu_data.angular_velocity.y = Angular_rate_Traverse_Y;
        imu_data.angular_velocity.z = Angular_rate_Down_Z;
        imu_pub.publish(imu_data);

        // rad data
        imu_data_rad.header.stamp = ros::Time::now();
        imu_data_rad.header.frame_id = "map";
        //Qua
        imu_data_rad.orientation.x = q_.x();
        imu_data_rad.orientation.y = q_.y();
        imu_data_rad.orientation.z = q_.z();
        imu_data_rad.orientation.w = q_.w();

        //linear_acceleration
        imu_data_rad.linear_acceleration.x = Longitudinal_accel_X;
        imu_data_rad.linear_acceleration.y = Traverse_accel_Y;
        imu_data_rad.linear_acceleration.z = Down_accel_Z;

        //angular_velocity
        imu_data_rad.angular_velocity.x = Angular_rate_Long_X * PI/180;
        imu_data_rad.angular_velocity.y = Angular_rate_Traverse_Y * PI/180;
        imu_data_rad.angular_velocity.z = Angular_rate_Down_Z * PI/180;
        imu_rad_pub.publish(imu_data_rad);

}

/**********************************************************************/
void processINSRMS( int length, char *pData )
/**********************************************************************/
{
        unsigned short GPS_week_number;
        unsigned int GPS_time_ms;
        char IMU_alignment_status;
        char GPS_quality_indicator;
        float North_Position_RMS;
        float East_Position_RMS;
        float Down_Position_RMS;
        float North_Velocity_RMS;
        float East_Velocity_RMS;
        float Down_Velocity_RMS;
        double Roll_RMS;
        double Pitch_RMS;
        double Heading_RMS;

        /****************************************************************************/
        /*                                                                          */
        /*    IMU Alignment Status provides the status of the BX/BD935 INS Solution */
        /*                                                                          */
        /*    0 - GPS Only                                                          */
        /*    1 - Coarse leveling                                                   */
        /*    2 - Degraded solution                                                 */
        /*    3 - Aligned                                                           */
        /*    4 - Full navigation mode                                              */
        /*                                                                          */
        /****************************************************************************/

        /****************************************************************************/
        /*                                                                          */
        /*    GPS quality indicator maps the internal rover fix to GNSS quality     */
        /*    (same as in the NMEA GGA Protocol Message).                           */
        /*                                                                          */
        /*  0 - Fix not available or invalid                                      */
        /*  1 - GPS SPS Mode, fix valid                                           */
        /*  2 - Differential GPS, SPS Mode, fix valid                             */
        /*  3 - GPS PPS Mode, fix valid                                           */
        /*  4 - Real Time Kinematic. System used in RTK mode with fixed integers  */
        /*  5 - Float RTK. Satellite system used in RTK mode, floating integers   */
        /*  6 - Estimated (dead reckoning) Mode                                   */
        /*  7 - Manual Input Mode                                                 */
        /*  8 - Simulator Mode                                                    */
        /*                                                                          */
        /****************************************************************************/

        printf( "  -------------- GsofType: 50 - INS Full Navigation RMS Info - Length (in Bytes):%d ------------ \n", length);
        printf( " \n " );

        GPS_week_number = getU16( &pData );
        GPS_time_ms =  (int) getU32( &pData );

        IMU_alignment_status = *pData++;
        GPS_quality_indicator = *pData++;

        North_Position_RMS = getFloat( &pData );
        East_Position_RMS = getFloat( &pData );
        Down_Position_RMS = getFloat( &pData );

        North_Velocity_RMS = getFloat( &pData );
        East_Velocity_RMS = getFloat( &pData );
        Down_Velocity_RMS = getFloat( &pData );

        Roll_RMS = (double) getFloat( &pData );
        Pitch_RMS = (double) getFloat( &pData );
        Heading_RMS = (double) getFloat( &pData );

        printf( "  GPS_Week:%d TOWms:%u IMU_Status:%02X \n", GPS_week_number, GPS_time_ms, IMU_alignment_status );
        printf( " \n " );
        printf( "  GPS Quality:%02X North Pos RMS:%.3lf East Pos RMS:%.3lf Down Pos RMS:%.3lf \n", GPS_quality_indicator,  North_Position_RMS, East_Position_RMS, Down_Position_RMS );
        printf( " \n " );
        printf( "  N Vel RMS:%.3lf E Vel RMS:%.3lf D Vel RMS:%.3lf \n", North_Velocity_RMS, East_Velocity_RMS, Down_Velocity_RMS );
        printf( " \n " );
        printf( "  Roll RMS:%.3lf Pitch RMS:%.3lf Heading RMS:%.3lf \n", Roll_RMS, Pitch_RMS, Heading_RMS );
        printf( " \n " );
        printf( " -------------------------------------End INS Type 50------------------------- \n ");
}

/***********************************************************************
 * End of the GSOF subtype parsers.
 **********************************************************************
 */



/**********************************************************************/
void processGsofData( char* gsofData )
/**********************************************************************/
/* Called when a complete set of GSOF packets has been received.
 * The data bytes collected are avialble in global gsofData and the
 * number of those bytes is in gsofDataIndex.
 *
 * This routine just goes through the bytes and parses the sub-type
 * records.  Each of thos has a Type and a Length.  If the type is
 * one of the special types we know about, we call the proper parser.
 * Otherwise we just hex-dump the record.
 */
{
        int i;
        int gsofType;
        int gsofLength;
        char * pData;

        printf( "\n  GSOF Records\n" );
        pData = gsofData;

        while (pData < gsofData + gsofDataIndex )
        {
                gsofType   = *pData++;
                gsofLength = *pData++;

                // If the type is one that we know about, then call the specific
                // parser for that type.
                if ( gsofType == 1 )
                {
                        processPositionTime( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 2 )
                {
                        processLatLonHeight( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 3 )
                {
                        processECEF( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 4 )
                {
                        processLocalDatum( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 8 )
                {
                        processVelocityData( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 9 )
                {
                        processPdopInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 49 )
                {
                        processINSFullNavigation( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 50 )
                {
                        processINSRMS ( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 13 )
                {
                        processBriefSVInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 16 )
                {
                        processUtcTime( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 33 )
                {
                        processAllBriefSVInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 34 )
                {
                        processAllDetailedSVInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 14 )
                {
                        processSvDetailedInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 27 )
                {
                        processAttitudeInfo( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 26 )
                {
                        processPositionTimeUtc( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 6 )
                {
                        processEcefDelta( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 7 )
                {
                        processTangentPlaneDelta( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                if ( gsofType == 40 )
                {
                        processLbandStatus( gsofLength, pData );
                        pData += gsofLength;
                }
                else
                {
                        // Not a type we know about.  Hex dump the bytes and move on.
                        printf( "  GsofType:%d  len:%d\n  ",
                                gsofType,
                                gsofLength
                                );

                        for ( i = 0; i < gsofLength; ++i )
                        {
                                printf( "%02X%s",
                                        *pData++,
                                        i % 16 == 15 ? "\n  " : " "
                                        );
                        }
                        // Terminate the last line if needed.
                        if (gsofLength % 16 != 0)
                        {
                          printf("\n");
                        }
                }

                printf( "\n" );
        }
        printf( "\n" );

} /* end of processGsofData() */



/**********************************************************************/
void postGsofData( char * pData, int length )
/**********************************************************************/
// Called whenever we get a new Trimcomm GSOF packet (type 0x40).
// These all contain a portion (or all) of a complete GSOF packet.
// Each portion contains a Transmission Number, an incrementing value
// linking related portions.
// Each portion contains a Page Index, 0..N, which increments for each
// portion in the full GSOF packet.
// Each portion contains a Max Page Index, N, which is the same for all
// portions.
//
// Each portion's data is appended to the global buffer, gsofData[].
// The next available index in that buffer is always gsofDataIndex.
// When we receive a portion with Page Index == 0, that signals the
// beginning of a new GSOF packet and we restart the gsofDataIndex at
// zero.
//
// When we receive a portion where Page Index == Max Page Index, then
// we have received the complete GSOF packet and can decode it.
{
        int a;
        int b;
        int c;
        int d;
        int gsofTransmissionNumber;
        int gsofPageIndex;
        int gsofMaxPageIndex;
        int i;

        // a = *pData++ ;
        // b = *pData++ ;
        // c = *pData++ ;
        // d = *pData++ ;
        pData = pData+4;
        gsofTransmissionNumber = *pData++;
        gsofPageIndex = *pData++;
        gsofMaxPageIndex = *pData++;

        printf( "  GSOF packet: Trans#:%d  Page:%d MaxPage:%d\n",
                gsofTransmissionNumber,
                gsofPageIndex,
                gsofMaxPageIndex
                );

        // If this is the first portion, restart the buffering system.
        if (gsofPageIndex == 0)
        {
          gsofDataIndex = 0;
        }

        // Transfer the data bytes in this portion to the global buffer.
        for (i = 7; i < length - 5; ++i)
        {
          gsofData[gsofDataIndex++] = *pData++;
        }

        // If this is the last portion in a packet, process the whole packet.
        if (gsofPageIndex == gsofMaxPageIndex)
        {
          processGsofData(gsofData);
        }

} /* end of postGsofData() */



/**********************************************************************/
int gc( void )
/**********************************************************************/
/* This is a getchar() wrapper.  It just returns the characters
 * from standard input.  If it detects end of file, it aborts
 * the entire program.
 *
 * This is ugly because if you are in the middle of a packet there is
 * no indication.  But what do you want from a simple diagnostic tool?
 */
{
        int c;

        c = getchar();
        if (c != EOF)
        {
          return c;
        }

        printf( "END OF FILE \n" );
        _exit( 0 );
} /* end of gc() */

/**********************************************************************/
void handle_udp_msg(int fd)
/**********************************************************************/
{
        char buf[BUFF_LEN];
        socklen_t len;
        int count;
        struct sockaddr_in clent_addr;
        ros::Rate loop_rate(50);

        while(ros::ok())
        {
                memset(buf, 0, BUFF_LEN);
                len = sizeof(clent_addr);
                count = recvfrom(fd, buf, BUFF_LEN, 0, (struct sockaddr*)&clent_addr, &len);
                if(count == -1)
                {
                        printf("recieve data fail!\n");
                        return;
                }
                // hahaha
                postGsofData( buf, count);
                // std::cout << buf[0] << std::endl;
                memset(buf, 0, BUFF_LEN);
                sprintf(buf, "I have recieved %d bytes data!\n", count);
                printf("  server:%s\n",buf);
                sendto(fd, buf, BUFF_LEN, 0, (struct sockaddr*)&clent_addr, len);
                ros::spinOnce();
                loop_rate.sleep();

        }
}

/**********************************************************************/
int main( int argc, char **argv )
// int main( int argn, char **argc )

/**********************************************************************/
/* Main entry point.  Looks for Trimcomm packets.  When we find one with
 * type 0x40, its bytes are extracted and passed on to the GSOF
 * handler.
 */
{
		initial_para();
        ros::init(argc, argv, "imu");
        ros::NodeHandle n;
        imu_pub = n.advertise<sensor_msgs::Imu>("imu_data", 20);
        imu_rad_pub = n.advertise<sensor_msgs::Imu>("imu_data_rad", 20);
        gnss_pub = n.advertise<geometry_msgs::PoseStamped>("gnss_data", 20);
        gnss2local_pub = n.advertise<geometry_msgs::PoseStamped>("gnss2local_data", 20);

        int server_fd, ret;
        struct sockaddr_in ser_addr;

        printf( " GSOF Parser\n");

        server_fd = socket(AF_INET, SOCK_DGRAM, 0); //AF_INET:IPV4;SOCK_DGRAM:UDP
        if(server_fd < 0)
        {
                printf("create socket fail!\n");
                return -1;
        }

        memset(&ser_addr, 0, sizeof(ser_addr));
        ser_addr.sin_family = AF_INET;
        ser_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        ser_addr.sin_port = htons(SERVER_PORT);

        ret = bind(server_fd, (struct sockaddr*)&ser_addr, sizeof(ser_addr));
        if(ret < 0)
        {
                printf("socket bind fail!\n");
                return -1;
        }

        handle_udp_msg(server_fd);

        close(server_fd);

        //int tcStx ;
        //int tcStat ;
        //int tcType ;
        //int tcLength ;
        //int tcCsum ;
        //int tcEtx ;
        //char tcData[256] ;
        //int i ;
        // while ( 1 )
        // {

        //   tcStx = gc() ;
        //   if ( tcStx == 0x02)
        //   {
        //     printf("I'm here ...\n");
        //     tcStat = gc() ;
        //     tcType = gc() ;
        //     tcLength = gc() ;
        //     for ( i = 0 ; i < tcLength ; ++i )
        //       tcData[i] = gc() ;

        //     tcCsum = gc() ;
        //     tcEtx = gc() ;
        //     printf( "STX:%02Xh  Stat:%02Xh  Type:%02Xh  "
        //             "Len:%d  CS:%02Xh  ETX:%02Xh\n",
        //             tcStx,
        //             tcStat,
        //             tcType,
        //             tcLength,
        //             tcCsum,
        //             tcEtx
        //           ) ;

        //     if (tcType == 0x40)
        //       postGsofData( tcData, tcLength ) ;
        //   }
        //   else {
        //     printf("Skipping %02X\n", tcStx ) ;
        //   }

        // }

        return 0;
} // main
