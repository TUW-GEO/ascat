#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:23:21 2021

@author: leonard
"""

import os
import unittest
from datetime import datetime, timedelta

import numpy as np

import sys
sys.path.append('..')

from src.ascat.download_connectors import HSAFConnector, EumetsatConnector
from src.ascat.download_interface import download_hsaf, download_eumetsat

from tempfile import mkdtemp

import configparser

class TestDownload(unittest.TestCase):
    
    def setUp(self):
        
        self.credential_file = 'test_credentials.ini'
        yesterday = (datetime.now() - timedelta(1)).strftime('%Y%m%d')
        today = datetime.today().strftime('%Y%m%d')
        self.start_date = yesterday
        self.end_date = today
        self.path = mkdtemp()
        
    def tearDown(self):
        
        self.credential_file = None
        self.start_date = None
        self.end_date = None
        
    def test_creds(self):
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        assert len(credentials.sections())>0
        
    def test_eumetsat_connect(self):
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = EumetsatConnector()
        connector.connect(credentials=credentials)
        
    def test_eumetsat_download(self):  
        
        
        product = "EO:EUM:DAT:METOP:SOMO12"
        coords = [[-1.0, -1.0],
                  [4.0, -4.0],
          		  [8.0, -2.0],
          		  [9.0, 2.0],
          		  [6.0, 4.0],
          		  [1.0, 5.0],
          		  [-1.0, -1.0]]
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = EumetsatConnector()
        connector.connect(credentials=credentials)
        
        connector.download(product=product,
                        download_dir=self.path,
                        coords=coords,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        file_limit=1)
    	
    def test_hsaf_connect(self):
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        connector.close()
        
    def test_hsaf_h08_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h08'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h10_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h10'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h16_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h16'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h101_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h101'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h102_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h102'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h103_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h103'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
    
    def test_hsaf_h104_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h104'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
    def test_hsaf_h105_download(self):
        
        
        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)
        
        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        
        product = 'h105'
        
        connector.download(product=product,
                           download_dir=self.path,
                           start_date=self.start_date,
                           end_date=self.end_date,
                           file_limit=1)
        
        connector.close()
        
        
if __name__ == '__main__':
    unittest.main()
        