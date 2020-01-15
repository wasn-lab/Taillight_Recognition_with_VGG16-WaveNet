how to use:

UdpClient UDPclient;
UDPclient.initial (GlobalVariable::UI_UDP_IP, GlobalVariable::UI_UDP_Port);

int sendnumber = UDPclient.send_obj_to_vcu (cur_cluster, cur_cluster_num);
int sendnumber = UDPclient.send_obj_to_server (cur_cluster, cur_cluster_num);
int sendnumber = UDPclient.send_obj_to_rsu (cur_cluster, cur_cluster_num);
    
cout << "[UDP]:" << sendnumber << endl;
    
    
CanModule::ReadAndWrite_controller (cur_cluster, cur_cluster_num);
    