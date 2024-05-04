import os 
import time 

from .sensor import VirtualGPS 

if __name__ == "__main__": 
    try: 
        my_gps = VirtualGPS() 
        my_gps.setup() 

        while True: 
            my_gps.read_gps_state() 
            # time.sleep(0.2) 
    except KeyboardInterrupt: 
        my_gps.terminate() # plot line chart 
        os._exit(0)
    except Exception as e: 
        print(e) 