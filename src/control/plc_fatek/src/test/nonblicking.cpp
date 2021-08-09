#include <unistd.h> // read()
#include <termios.h>
#include <stdio.h>

    
class keyboard{
    public:
        keyboard();
        ~keyboard();
        int kbhit();
        int getch();

    private:
        struct termios initial_settings, new_settings;
        int peek_character;
};

keyboard::keyboard(){
    tcgetattr(0,&initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag &= ~ICANON;
    new_settings.c_lflag &= ~ECHO;
    new_settings.c_lflag &= ~ISIG;
    new_settings.c_cc[VMIN] = 1;
    new_settings.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
    peek_character=-1;
}
    
keyboard::~keyboard(){
    tcsetattr(0, TCSANOW, &initial_settings);
}
    
int keyboard::kbhit(){
    unsigned char ch;
    int nread;
    if (peek_character != -1) return 1;
    new_settings.c_cc[VMIN]=0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0,&ch,1);
    new_settings.c_cc[VMIN]=1;
    tcsetattr(0, TCSANOW, &new_settings);

    if (nread == 1){
        peek_character = ch;
        return 1;
    }
    return 0;
}
    
int keyboard::getch(){
    char ch;

    if (peek_character != -1){
        ch = peek_character;
        peek_character = -1;
    }
    else read(0,&ch,1);
    return ch;
}



int main(){
    int key_nr;
    char key;
    keyboard keyb;

    while(true){
        if( keyb.kbhit() ){
            key_nr = keyb.getch(); //return int
            key = key_nr; // get ascii char
            // do some stuff
            printf("%c",key);
            if(key=='0') break;
        }

    }
    return 0;
}