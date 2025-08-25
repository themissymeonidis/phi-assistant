import datetime
import json

class Tools:
    def __init__(self):
        pass

    def get_current_time(self):
        result = {
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result