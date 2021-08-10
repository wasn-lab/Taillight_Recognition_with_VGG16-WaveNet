#include <ros/ros.h> 
#include <serial/serial.h>  //ROS serial package 
#include <std_msgs/String.h> 
#include <std_msgs/Empty.h>

#include "msgs/BackendInfo.h"

const int NumOfTopic = 8;

class PLCControl{
    
    public:
        PLCControl(std::string);
        ~PLCControl();

    private:
        /*parameter*/
        ros::NodeHandle pnh_;

        ros::Publisher read;
        ros::Subscriber write;


        msgs::BackendInfo msg_Backend;    
        ros::Publisher Publisher_Backend;

        serial::Serial ser;
        int ctrl;

        void write_callback(const std_msgs::String::ConstPtr&);

        /*fuction*/
        void run();
        std::string readMsg(std::string);
        void sendMsg1(const std::string);
        void sendMsg2(const std::string);
        void sendMsg3(const std::string);
        void sendMsg4(const std::string);
        void sendMsg5(const std::string);
        void sendMsg7(const std::string);
        char DecToHex (int);
        void HexToBin (char, int*);
        std::string generateChecksum(std::string);


};

void PLCControl::write_callback(const std_msgs::String::ConstPtr& msg) {
    std::string ss = msg->data;
    ctrl = ss[0]-48;
    //std::cout<<ctrl<<std::endl;
    ss = ss.substr(1);
    
    if(ctrl == 6) Publisher_Backend.publish(msg_Backend);
    else{
        if(ser.isOpen()){ 
            //ROS_INFO_STREAM("Writing to serial port: " << ss); 
            ser.write(ss);   //send serial data
        }
        else{
            //ROS_INFO_STREAM("Writing to serial port: " << msg->data);
            ROS_INFO_STREAM("Serialport is unavailable.\n");
        }
    }
    
} 

PLCControl::PLCControl(std::string portpath){

    // Subscriber-------------------
    write = pnh_.subscribe("/control/plc_write", 10, &PLCControl::write_callback, this); 

    // Publisher--------------------
    read = pnh_.advertise<std_msgs::String>("/control/plc_info", 10);

    Publisher_Backend = pnh_.advertise<msgs::BackendInfo>("Backend/Info", 1);

    //Initial
    // Set up & Open serial port----
    try 
    {  
        ser.setPort(portpath); 
        ser.setBaudrate(9600);
        serial::Timeout to = serial::Timeout::simpleTimeout(5000); 
        ser.setTimeout(to); 
        ser.open(); 
    }
    catch (serial::IOException& e) 
    {   
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        ROS_ERROR_STREAM("Unable to open port "); 
        //return -1; 
    } 

    // Check serial port------------
    if(ser.isOpen()) 
    { 
        ROS_INFO_STREAM("Serial Port initialized"); 
    } 
    else 
    { 
        //return -1; 
    } 

    run();
}

PLCControl::~PLCControl(){}

void PLCControl::run(){

    ros::Rate loop_rate(10); 

    //ctrl =0;

    while(ros::ok()) 
    { 
        //std::cout << ser.available() << std::endl;
        if(ser.available()){ 
            std::string result;
            std_msgs::String state;

            result = ser.read(ser.available());
            //ROS_INFO_STREAM("Reading from serial port:");
            //std::cout<<result<<std::endl;
            result = readMsg(result);
            //std::cout<<result<<std::endl;
            //sendMsg2(result);
            
            switch(ctrl){
                case 1:
                    sendMsg1(result);
                    state.data = result;
                    read.publish(state);
                    break;
                case 2:
                    sendMsg2(result);
                    break;
                case 3:
                    sendMsg3(result);
                    break;
                case 4:
                    sendMsg4(result);
                    break;
                case 5:
                    sendMsg5(result);
                    break;
                case 6:
                    //std::cout<<"publish!"<<std::endl;
                    //Publisher_Backend.publish(msg_Backend);
                    break;
                case 7:
                    sendMsg7(result);
                    break;
                default:
                    sendMsg1(result);
                    break;
            }
            //Publisher_Backend.publish(msg_Backend);

            //sendMsg2(result);
            //std::cout << result << std::endl;
            //ROS_INFO_STREAM("Read: " << result.data);
            //read.publish(result); 
            //usleep(2000000);
        } 

        ros::spinOnce(); 
        loop_rate.sleep(); 

    } 

}

std::string PLCControl::readMsg(std::string data){
    //if((data[0] != 2) && (data[data.length()-1] != 3))
    //    return data;
    //cmd = data.substr(1,2);
    //std::string msg;

    std::string msg = data.substr(6);
    //msg[data.length()-5] = '\0';
    msg = msg.substr(0,msg.length()-3);
    //std::cout<<msg<<std::endl;
    return msg;
}

void PLCControl::sendMsg1(const std::string msg){
    int bin[4];
    /*
    for(int i = 0; i < msg.length(); i++){
        HexToBin(msg[i],bin);
        for(int j=0; j<4; j++){
            std::cout << bin[j] <<" ";
        }
    }

    std::cout<<std::endl;
    */
    //publish data
    HexToBin(msg[1],bin);
    msg_Backend.ACC_state=bin[2];
    msg_Backend.gross_power=bin[1];
    msg_Backend.indoor_light=bin[0];
    HexToBin(msg[2],bin);
    msg_Backend.hand_brake=bin[3];
    msg_Backend.wiper=bin[2];
    msg_Backend.estop=bin[1];
    msg_Backend.air_conditioner=bin[0];
    HexToBin(msg[3],bin);
    msg_Backend.right_turn_light=bin[3];
    msg_Backend.left_turn_light=bin[2];
    msg_Backend.headlight=bin[1];
    msg_Backend.door=bin[0];

}

void PLCControl::sendMsg2(const std::string msg){
    float data[2];
    int n;
    //std::cout << msg << std::endl;
    for(int i = 0; i < msg.length() ; i += 4){
        data[i/4] = 0.0;
        //std::cout << i/4 << std::endl;
        for(int j=i ; j < (i+4) ; j++){
            if((int)msg[j] < 58){
                n = msg[j]-48;
            }
            else{
                n = msg[j]-65+10; 
            }
            //std::cout << n << " ";
            data[i/4] *= 16;
            data[i/4] += n; 
        }
        //std::cout << data[i/4] << std::endl;    
    }
    //std::cout<<std::endl;
    
    //publish data
    float x;
    x = data[1] + data[0] * 1000;
    //std::cout << x << std::endl;
    msg_Backend.odometry = x;
    msg_Backend.mileage = x;
    
}

void PLCControl::sendMsg3(const std::string msg){
    float data[5];
    int n;
    //std::cout << msg << std::endl;
    for(int i = 0; i < msg.length() ; i += 4){
        data[i/4] = 0.0;
        //std::cout << i/4 << std::endl;
        for(int j=i ; j < (i+4) ; j++){
            if((int)msg[j] < 58){
                n = msg[j]-48;
            }
            else{
                n = msg[j]-65+10; 
            }
            data[i/4] *= 16;
            data[i/4] += n; 
        }
        //std::cout << data[i/4] << std::endl;    
    }
    //std::cout<<std::endl;

    msg_Backend.speed = data[0];
    msg_Backend.air_pressure = data[1]*0.1;
    //msg_Backend. = data[2];
    //msg_Backend. = data[3];
    msg_Backend.gross_current = data[4];
    
}

void PLCControl::sendMsg4(const std::string msg){
    float data[5];
    int n;
    //std::cout << msg << std::endl;
    for(int i = 0; i < msg.length() ; i += 4){
        data[i/4] = 0.0;
        //std::cout << i/4 << std::endl;
        for(int j=i ; j < (i+4) ; j++){
            if((int)msg[j] < 58){
                n = msg[j]-48;
            }
            else{
                n = msg[j]-65+10; 
            }
            data[i/4] *= 16;
            data[i/4] += n; 
        }
        //std::cout << data[i/4] << std::endl;    
    }
    //std::cout<<std::endl;

    msg_Backend.highest_voltage = data[0]*0.01;
    msg_Backend.highest_number = data[1];
    msg_Backend.lowest_volage = data[2]*0.01;
    msg_Backend.lowest_number = data[3];
    msg_Backend.voltage_deviation = data[4]*0.01;
    
}

void PLCControl::sendMsg5(const std::string msg){
    float data[6];
    int n;
    //std::cout << msg << std::endl;
    for(int i = 0; i < msg.length() ; i += 4){
        data[i/4] = 0.0;
        //std::cout << i/4 << std::endl;
        for(int j=i ; j < (i+4) ; j++){
            if((int)msg[j] < 58){
                n = msg[j]-48;
            }
            else{
                n = msg[j]-65+10; 
            }
            data[i/4] *= 16;
            data[i/4] += n; 
        }
        //std::cout << data[i/4] << std::endl;    
    }
    //std::cout<<std::endl;

    msg_Backend.highest_temperature = data[0]*0.001;
    msg_Backend.highest_temp_location = data[1];
    msg_Backend.gross_voltage = data[2];
    //msg_Backend. = data[3];
    msg_Backend.battery = data[4];
    msg_Backend.motor_temperature = data[5]*0.1;
    
}

void PLCControl::sendMsg7(const std::string msg){
    std::cout<<msg;
}

char PLCControl::DecToHex (int n){

    char hex[] = {'0', '1', '2', '3','4','5'
                 ,'6','7','8','9','A','B','C'
                 ,'D','E','F'
                 };
    
    return hex[n];
}

void PLCControl::HexToBin (char c, int* bin){
    int n;
    //std::cout<<(int)c<<std::endl;
    if((int)c <= 57){
        n = c-48;
    }
    else{
        n = c-65+10; 
    }
    //std::cout << n << std::endl;
    for(int i = 0; i < 4; i++){
            bin[i] = (n%2);
            n = n/2;
    }
}

std::string PLCControl::generateChecksum(std::string str){
    int sum =0;
    for(int i = 0; i <  str.length(); i++){
        sum += str[i];
    }

    sum +=2;
    sum = sum % 256;
    std::string out;
    out = DecToHex(sum/16);
    out = out + DecToHex(sum%16);

    return out;
}

int main (int argc, char** argv){

    ros::init(argc, argv, "plc_serial_ctrl");
    std::string portpath = "/dev/ttyS0";
    PLCControl obj(portpath);

    ros::spin();

    return 0;
} 