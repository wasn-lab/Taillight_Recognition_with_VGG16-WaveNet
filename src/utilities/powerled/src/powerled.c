//gcc -o powerled powerled.c -lorcania -lfdcore -lfdproto -Wall

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <pthread.h>



/*The Text Data*/
struct text_data
{
	int index;
	char *text_data;
	int text_data_len;
};

struct text_data_list
{
	struct text_data *node;
	struct text_data_list *next;
};

void load_powerled_config(void);
void load_text_data(int select_mode);
void send_text_data(int select_mode);
//void atoh(char *ascii_ptr, uint8_t *hex_ptr, int len);
void atoh(char *ascii_ptr, char *hex_ptr, int len);
int text_data_list_add(struct text_data *new_text_data, int select_mode);

struct text_data_list *text_data_list_head_1 = NULL;
struct text_data_list *text_data_list_head_2 = NULL;

char powerled_ip[16];
int powerled_port;
int sleep_time;

void _show_text_data_list(void)
{
	struct text_data_list *temp;
	
	temp = text_data_list_head_1;
	while(temp->next != NULL)
	{
		temp = temp->next;
		printf("index: %d\n", temp->node->index);
		for(int i = 0; i < temp->node->text_data_len; i++)
			printf("%hhx ", temp->node->text_data[i]);
		printf("\n");
	}

	temp = text_data_list_head_2;
	while(temp->next != NULL)
	{
		temp = temp->next;
		printf("index: %d\n", temp->node->index);
		for(int i = 0; i < temp->node->text_data_len; i++)
			printf("%hhx ", temp->node->text_data[i]);
		printf("\n");
	}	
}

void send_text_data(int select_mode)
{
	struct sockaddr_in si_other;
	int s, slen = sizeof(si_other);
	struct text_data_list *send_node;
	
	if((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
	{
		perror("socket error");
		exit(1);
	}

	memset((char *) &si_other, 0, sizeof(si_other));
	si_other.sin_family = AF_INET;
	si_other.sin_port = htons(powerled_port);
	
	if(inet_aton(powerled_ip , &si_other.sin_addr) == 0){
		fprintf(stderr, "inet_aton() failed\n");
		exit(1);
	}	

	if(select_mode == 1)
		send_node = text_data_list_head_1;
	else
		send_node = text_data_list_head_2;
	
	while(send_node->next != NULL)
	{
		send_node = send_node->next;
		
		if(sendto(s, send_node->node->text_data, send_node->node->text_data_len, 0 , (struct sockaddr *) &si_other, slen) == -1)
		{
			perror("sendto()");
			exit(1);
		}
		printf("send text data list 1 index = %d, len = %d\n", send_node->node->index, send_node->node->text_data_len);

		usleep(1000 * sleep_time);	
	}	
}

int main(int argc, char *argv[])
{
	int select_mode;

	if(argc != 2){ 
		printf("./powerled <select_mode>\n");
		printf("-- usage --\n");
		printf("<select_mode> 1: hand, 2: self\n");
        printf("-- example --\n");
		printf("./powerled 1\n");
		return 1;
	}

	select_mode = atoi(argv[1]);
	printf("select mode: %d\n", select_mode);

	//Creation of text_data_list
	text_data_list_head_1 = malloc(sizeof(struct text_data_list));
	text_data_list_head_1->next = NULL;
	if(text_data_list_head_1 == NULL)
	{
		printf("[ERROR] no memory to init text_data_list_head \n");
		exit(1);
	}

	text_data_list_head_2 = malloc(sizeof(struct text_data_list));
	text_data_list_head_2->next = NULL;
	if(text_data_list_head_2 == NULL)
	{
		printf("[ERROR] no memory to init text_data_list_head \n");
		exit(1);
	}

	load_powerled_config();
	load_text_data(1);
	load_text_data(2);
	_show_text_data_list();

	send_text_data(select_mode);

	return 0;
}

void load_powerled_config(void)
{
	FILE *fp_powerled_config;
	char temp_buff[30];
	int ret;

	//get parameters from tsc_config.txt
	memset(powerled_ip, '\0', 16*sizeof(char));
	powerled_port = 0;
	sleep_time = 1;

	if((fp_powerled_config = fopen("powerled_config.txt","r")) == NULL)
	{
		printf("open file powerled_config.txt error....\n");
		exit(1);
	}
	
	while(!feof(fp_powerled_config)){
		ret = fscanf(fp_powerled_config, "%s", temp_buff);
		if(ret < 0){
			printf("fscanf file powerled_config.txt error....\n");
			exit(1);
		}

		if(strcmp(temp_buff, "POWERLED_IP") == 0){
			ret = fscanf(fp_powerled_config, "%s", powerled_ip);
			if(ret < 0){
				printf("fscanf file powerled_config.txt error....\n");
				exit(1);
			}
		}else if(strcmp(temp_buff, "POWERLED_PORT") == 0){
			ret = fscanf(fp_powerled_config, "%d", &powerled_port);
			if(ret < 0){
				printf("fscanf file powerled_config.txt error....\n");
				exit(1);
			}
		}else if(strcmp(temp_buff, "SEND_INTERVAL") == 0){
			ret = fscanf(fp_powerled_config, "%d", &sleep_time);
			if(ret < 0){
				printf("fscanf file powerled_config.txt error....\n");
				exit(1);
			}
			break;
		}else{
			printf("Unknown parameter in powerled_config.txt: %s\n", temp_buff );
		}
	}

	printf("POWERLED IP: %s\n", powerled_ip );
	printf("POWERLED PORT: %d\n", powerled_port);
	printf("SEND INTERVAL: %d\n", sleep_time);

	fclose(fp_powerled_config);
}

void load_text_data(int select_mode)
{
	FILE *fp_text_data;
	char buff[3500];
	char buff_2[3500];
	char ascii_char[2], temp_char;
	unsigned int index, i, result;
	struct text_data *new_text_data;
	int index_count;

	//get text data from self_data.txt or hand_data.txt
	if(select_mode == 1)
	{
		if((fp_text_data = fopen("hand_data.txt","r")) == NULL)
		{
			printf("open file hand_data.txt error....\n");
			exit(1);
		}
	}else{
		if((fp_text_data = fopen("self_data.txt","r")) == NULL)
		{
			printf("open file self_data.txt error....\n");
			exit(1);
		}
	}	

	index_count = 0;
	memset(buff, '\0', 3500*sizeof(char));
	memset(buff_2, '\0', 3500*sizeof(char));
	while(fscanf(fp_text_data, "%[^\n]%*c", buff) != EOF){
		index_count++;
		index = 0;
		i = 0;
		memset(buff_2, '\0', 3500*sizeof(char));
		
		while(i < strlen(buff))
		{
			if((i == (strlen(buff) - 3)) || (buff[i+1] == ' '))
			{
				ascii_char[0] = '0';
				ascii_char[1] = buff[i];
				i = i + 2;				
			}else if(buff[i+1] != ' ')
			{
				ascii_char[0] = buff[i];
				ascii_char[1] = buff[i+1];
				i = i + 3;				
			}else{
				if(select_mode == 1)
					printf("hand_data.txt format error....\n");
				else
					printf("self_data.txt format error....\n");
				exit(1);
			}

			if(i < strlen(buff))
			{
				atoh(ascii_char, &temp_char, 2);
				//printf("%hhx ", temp_char);
				buff_2[index] = temp_char;
				index++;
			}			
		}
		//printf("\n");

		new_text_data = malloc(sizeof(struct text_data));
		if(new_text_data == NULL)
		{
			printf("[ERROR] no memory to malloc text_data\n");			
			exit(1);
		}

		new_text_data->index = index_count;
		new_text_data->text_data_len = index;
		
		new_text_data->text_data = malloc(new_text_data->text_data_len * sizeof(char));
		if(new_text_data->text_data == NULL)
		{
			printf("[ERROR] no memory to malloc new_text_data->text_data\n");
			exit(1);
		}
		memcpy(new_text_data->text_data, buff_2, new_text_data->text_data_len);

		/*
		for(i = 0; i < new_traffic_signal_data->ts_data_len; i++)
			printf("%hhx ", new_traffic_signal_data->ts_data[i]);
		printf("\n");*/

		result = text_data_list_add(new_text_data, select_mode);
		if(result != 0){
			printf("[ERROR] failed to add  new_text_data.\n");
			exit(1);
		}
		
		memset(buff, '\0', 3500*sizeof(char));		
	}	
}

int text_data_list_add(struct text_data *new_text_data, int select_mode)
{
	printf("[LOG] Prepare to add text_data node into list.\n");

	struct text_data_list *ptr;
	struct text_data_list *temp;
	int result = 0;
    
	ptr = malloc(sizeof(struct text_data_list));
	if(ptr == NULL)
	{
		printf("[ERROR] no memory to add new text_data_list \n");
		result = -1;
		goto out;
	}
    
	ptr->node = new_text_data;
	ptr->next = NULL;

	if(select_mode == 1)
    	temp = text_data_list_head_1;
	else
		temp = text_data_list_head_2;
	
	while(temp->next != NULL)
	{
		temp = temp->next;
	}
	temp->next = ptr;
    
out:
    return result;
}

//void atoh(char *ascii_ptr, uint8_t *hex_ptr, int len)
void atoh(char *ascii_ptr, char *hex_ptr, int len)
{
	int i;

	for(i = 0; i < (len / 2); i++)
	{
		*(hex_ptr+i)   = (*(ascii_ptr+(2*i)) <= '9') ? ((*(ascii_ptr+(2*i)) - '0') * 16 ) : (((*(ascii_ptr+(2*i)) - 'a') + 10) << 4);
		*(hex_ptr+i)  |= (*(ascii_ptr+(2*i)+1) <= '9') ? (*(ascii_ptr+(2*i)+1) - '0') : (*(ascii_ptr+(2*i)+1) - 'a' + 10);
	}
}

