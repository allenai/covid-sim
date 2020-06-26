#!/bin/bash
DATASET_NAME="${DATASET_NAME:-data}"
DATASET_URL="${DATASET_URL:-https://storage.googleapis.com/skiff-models/covid-sim/demo-data-partial/demo_data_01.zip}"
LOCAL_DATASET_DIR=/usr/src/app/covid-ai2

echo "INFO: Downloading dataset from ${DATASET_URL})..."
wget ${DATASET_URL} || exit 1
unzip demo_data.zip -d ${LOCAL_DATASET_DIR}/${DATASET_NAME}

streamlit run demo.py --server.enableCORS false
