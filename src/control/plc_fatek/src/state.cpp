#include <ros/ros.h> 
#include <serial/serial.h>  //ROS serial package 
#include <std_msgs/String.h> 
#include <std_msgs/Empty.h> 

bool con = false;
std::string commond;

struct State{
  /* data */
  int ACC_state;
  int gross_power;
  int indoor_light;

  //int hand_brake;
  int wiper;
  //int estop;
  int air_conditioner;

  int right_turn_light;
  int left_turn_light;
  int headlight;
  int door;

  State& operator=(const State& a){
    
    ACC_state = a.ACC_state;
    gross_power = a.gross_power;
    indoor_light = a.indoor_light;
    wiper = a.wiper;
    air_conditioner = a.air_conditioner;
    right_turn_light = a.right_turn_light;
    left_turn_light = a.left_turn_light;
    headlight = a.headlight;
    door = a.door;

    return *this;
  }

}state, control;

void HexToBin (char c, int* bin){
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

char DecToHex (int n)
{
    char hex[] = {'0', '1', '2', '3','4','5'
                 ,'6','7','8','9','A','B','C'
                 ,'D','E','F'
                 };
    
    return hex[n];
}

std::string generateChecksum(std::string str){
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

std::string StatetoIns(State s){
  std::string msg = "034701R029670";
  int x;
  x = s.ACC_state*4 + s.gross_power*2 + s.indoor_light;
  msg = msg + DecToHex(x);
  x = s.wiper*4 + s.air_conditioner;
  msg = msg + DecToHex(x);
  x = s.right_turn_light*8 + s.left_turn_light*4 + s.headlight*2 + s.door;
  msg = msg + DecToHex(x);

  return msg;
}

void show(){
  printf("\033c");
  std::cout << std::endl;
  std::cout<<"1: door             : " << state.door << std::endl;
  std::cout<<"2: head light       : " << state.headlight << std::endl;
  std::cout<<"3: left turn light  : " << state.left_turn_light << std::endl;
  std::cout<<"4: right turn light : " << state.right_turn_light << std::endl;
  std::cout<<"5: air conditioner  : " << state.air_conditioner << std::endl;
  std::cout<<"6: wiper            : " << state.wiper << std::endl;
  std::cout<<"7: indoor light     : " << state.indoor_light << std::endl;
  std::cout<<"8: gross power      : " << state.gross_power << std::endl;
  std::cout<<"9: ACC state        : " << state.ACC_state << std::endl;
}

void read_callback(const std_msgs::String::ConstPtr& msg){

  //ROS_INFO("I heard: [%s]", msg->data.c_str());
  std::string ss = msg->data;

  int bin[4];
  HexToBin(ss[1],bin);
  state.ACC_state = bin[2];
  state.gross_power = bin [1];
  state.indoor_light = bin[0];
  HexToBin(ss[2],bin);
  state.wiper = bin[2];
  state.air_conditioner = bin[0];
  HexToBin(ss[3],bin);
  state.right_turn_light = bin[3];
  state.left_turn_light = bin[2];
  state.headlight = bin[1];
  state.door = bin[0];

}

void control_callback(const std_msgs::String::ConstPtr& msg){
  commond = msg->data;
  //std::cout << commond << std::endl;
  char com1 = commond[0];
  int com2 = commond[2] - 48;
  //std::cout << com1 << " " << com2 << std::endl;
  if(com2 != 0 && com2 != 1) com1 = '0';

  control = state;

  switch(com1){
    case '1':
      control.door = com2;
      break;
    case '2':
      control.headlight = com2;
      break;
    case '3':
      control.left_turn_light = com2;
      break;
    case '4':
      control.right_turn_light = com2;
      break;
    case '5':
      control.air_conditioner = com2;
      break;
    case '6':
      control.wiper = com2;
      break;
    case '7':
      control.indoor_light = com2;
      break;
    case '8':
      control.gross_power = com2;
      break;
    case '9':
      control.ACC_state = com2;
      break;
    default:

      break;
  };

  commond = StatetoIns(control);
  

  con = true;
}

std_msgs::String sendCommand(std::string ss, char ctrl){
  
  std_msgs::String msg;
  std::string check = generateChecksum(ss);
  char start = 2 , end = 3;

  msg.data =  ctrl + (start + ss + check + end);
  //ROS_INFO("You will publish %s", msg.data.c_str());   
  
  return msg;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "talker_state");  

  ros::NodeHandle n, param_n;     


  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("/control/plc_write", 1000);  
  
  ros::Subscriber read = n.subscribe("/control/plc_info", 1000, read_callback);

  ros::Subscriber control = n.subscribe("/control/plc_control", 1000, control_callback);

  int update_rate;
  param_n.getParam("update_rate", update_rate);

  ros::Rate loop_rate(10); 
  int count = 0;

  while (ros::ok())
  {
    show();

    std_msgs::String msg;

    std::string ss;


    ss = "034601R02966";
    msg = sendCommand(ss,49);
    chatter_pub.publish(msg); 
    usleep(update_rate);
    

    ss = "034602R02769";
    msg = sendCommand(ss,50);   
    chatter_pub.publish(msg);
    usleep(update_rate);
    
    ss = "034605R02785";
    msg = sendCommand(ss,51); 
    chatter_pub.publish(msg);
    usleep(update_rate);
    
    ss = "034605R02790";
    msg = sendCommand(ss,52);
    chatter_pub.publish(msg);
    usleep(update_rate);

    ss = "034606R02795";
    msg = sendCommand(ss,53);
    chatter_pub.publish(msg);
    usleep(update_rate);

    ss = "6";
    msg.data = ss ;
    chatter_pub.publish(msg);

    if(con){

      usleep(update_rate);

      std::string check = generateChecksum(commond);
      char start = 2 , end = 3, ctrl = 55;

      commond =  ctrl + (start + commond + check + end);
      msg.data = commond;
      chatter_pub.publish(msg);
      con = false;
    }


    ros::spinOnce();
    loop_rate.sleep();

  }
  return 0;
}