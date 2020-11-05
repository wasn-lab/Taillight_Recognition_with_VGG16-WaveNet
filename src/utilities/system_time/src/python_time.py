import getpass
import time

if __name__ == '__main__':
  username = getpass.getuser()

  while True:
    print( username + " - time now: " + str(time.time()))
    time.sleep(0.033) #fps 30
