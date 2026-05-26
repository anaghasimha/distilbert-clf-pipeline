# Enterprise DistilBERT Text Classification Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.44.0-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready, configuration-driven Natural Language Processing pipeline featuring a fine-tuned DistilBERT transformer model for sequence-level classification tasks. Engineered with strict software engineering discipline, this repository showcases decoupled modular application patterns, deterministic caching architectures, and robust MLOps container deployment isolation.

---

## 🏗️ System Architecture & Data Flow

This ecosystem intentionally decouples the **Configuration State**, **Data Lifecycles**, **Model Computations**, and the **User Interface Layer**. Components interact strictly via immutable data schema structures and file path tracking parameters.

```mermaid
graph TD
    %% Configuration Framework
    Config[config.yaml] -->|Provides Parameters| Utils[src/utils/common.py]
    
    %% Ingestion Execution
    DataRaw[data/raw/dataset.csv] --> Ingest[src/components/data_ingestion.py]
    Utils --> Ingest
    Ingest -->|Stratified Split| DataSplit[artifacts/data_ingestion/]
    
    %% Transformation Engine
    DataSplit --> Trans[src/components/data_transformation.py]
    Utils --> Trans
    Trans -->|Custom PyTorch Dataset| TrainLoop[src/components/model_trainer.py]
    
    %% Training Execution
    Utils --> TrainLoop
    TrainLoop -->|Native Loop + AMP| SavedWeights[artifacts/model_trainer/distilbert_clf_model/]
    
    %% Serving Engine
    SavedWeights --> PredPipe[src/pipeline/prediction_pipeline.py]
    Utils --> PredPipe
    PredPipe -->|Cached Singleton Engine| UI[src/ui/app.py]
    
    %% Styling and Branding Classes
    classDef config fill:#f9f,stroke:#333,stroke-width:2px;
    classDef components fill:#bbf,stroke:#333,stroke-width:1px;
    classDef storage fill:#ffb,stroke:#333,stroke-width:1px;
    
    class Config config;
    class Ingest,Trans,TrainLoop,PredPipe components;
    class DataRaw,DataSplit,SavedWeights storage;