import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("diabetes_model.joblib")

