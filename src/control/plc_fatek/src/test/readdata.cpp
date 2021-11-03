#include <iostream>
#include <cstring>

using namespace std;

char DecToHex (int n)
{
    char hex[] = {'0', '1', '2', '3','4','5'
                 ,'6','7','8','9','A','B','C'
                 ,'D','E','F'
                 };
    
    return hex[n];
}

string F1(string str){
    int sum =2;
    for(int i = 0; i <  str.length(); i++){
        sum += str[i];
    }

    sum = sum % 256;
    string out;
    out = DecToHex(sum/16);
    out = out + DecToHex(sum%16);

    return out;
}


int main (void){
    string str;
    
    while(1){
        cout<<"Input:";
        int sum = 0;
        cin>>str;
        if (str == "run") str =  "01411";
        if (str == "state") str = "0140";
        string out;
        out = F1 (str);

        cout<<"Output:"<<out<<endl;
        string output;
        char start = 2, end = 3;
        output = start + str + out + end;
        cout<<"MSG:"<<output<<endl;

    }
    return 0;
}