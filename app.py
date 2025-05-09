import streamlit as st
import pandas as pd
import numpy as np
import joblib


with open("diabetes_model.joblib", 'rb') as file:
    model = joblib.load(file)
