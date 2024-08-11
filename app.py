import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import requests

model = YOLO("./Models/advanced_shapes_model.h5")