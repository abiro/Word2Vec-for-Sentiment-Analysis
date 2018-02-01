import logging


format_str = ('[%(asctime)s - %(levelname)s:%(filename)s:%(funcName)s:' +
              '%(lineno)s] %(message)s')
logging.basicConfig(level=logging.INFO, format=format_str)
