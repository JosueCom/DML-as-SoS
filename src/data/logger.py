import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

"""
Author: Josue N Rivera
"""
class Logger():
    
    def __init__(self, 
                 log_keys:List, 
                 config:Optional[Dict]) -> None:
        self.options = config['save']
        self.logs = {
            '__prints__': [],
            '__configuration__': config
        }
        for key in log_keys:
            self.logs[key] = []

        dateinfo = datetime.now()
        self.filename = self.options['format']\
                            .replace('MONTH', str(dateinfo.month))\
                            .replace('DAY', str(dateinfo.day))\
                            .replace('YEAR', str(dateinfo.year))\
                            .replace('START_HOUR', str(dateinfo.hour))\
                            .replace('START_MINUTE', str(dateinfo.minute))

        self.logs['__dateinfo__'] = {
            'MONTH': dateinfo.month,
            'DAY': dateinfo.day,
            'YEAR': dateinfo.year,
            'START_HOUR': dateinfo.hour,
            'START_MINUTE': dateinfo.minute
        }

        self.print(f"Log started at: {dateinfo}")
    
    def append(self, 
               key:str, 
               item:Any) -> None:
        self.logs[key] = item

    def log(self, 
            key:str, 
            item:Any) -> None:
        self.logs[key].append(item)

    def print(self, 
              txt:str, screenon=True) -> None:
        self.log('__prints__', str(datetime.now()) + ': ' + txt)

        if (not self.options['printless']) and screenon:
            print(txt)

    def close(self) -> None:

        dateinfo = datetime.now()
        self.filename = self.filename.replace('END_HOUR', str(dateinfo.hour))\
                        .replace('END_MINUTE', str(dateinfo.minute))

        self.logs['__dateinfo__']['END_HOUR'] = dateinfo.hour
        self.logs['__dateinfo__']['END_MINUTE'] = dateinfo.minute

        self.print(f"Log closed at: {dateinfo}")
        self.print(f"Selected filename format: \'{self.filename}\'")

        if self.options['logs']:
            fn = os.path.join(self.options['path'], 'logs', self.filename.replace('TYPE', 'LOGS')) + '.json'
            
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            
            self.print(f"Data logs saved to \"{fn}\"")

            with open(fn, 'w') as f:
                json.dump(self.logs, f)


class LogLoad():

    def __init__(self,
                 log_file: str) -> None:

        with open(log_file) as f:
            self.logs = json.load(f)

        self.options = self.logs['__configuration__']['save']
        
    def retrieve_log(self,
                 key: str) -> List:

        return self.logs[key]
        
