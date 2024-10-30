import pandas as pd

class PreProcessor():
    def __init__(self, filepath):
        self.data_path = filepath
        self.df = None

