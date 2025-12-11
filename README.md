# Melanoma Detection - Medical Image Classification

Проект по классификации меланомы (рак кожи) на dermoscopy изображениях с использованием глубокого обучения.

## 📋 Описание

Система использует:

- **Dataset**: ISIC 2020 Training (33,126 изображений)
- **Model**: ResNet50 pre-trained на ImageNet
- **Framework**: PyTorch
- **Task**: Binary classification (Melanoma vs Benign)

## 🎯 Цель

Создать production-ready модель для классификации меланомы на медицинских снимках кожи с высокой точностью и возможностью развертывания.

## 📊 Данные

- **Total images**: 33,126
- **Melanoma**: 584 (1.76%)
- **Benign**: 32,539 (98.24%)
- **Source**: [ISIC Archive](https://www.isic-archive.com/)

⚠️ **Важно**: Dataset не включен в репозитории. Скачайте отдельно и положите в `data/isic/`

## 🏗️ Структура проекта

medical-melanoma-detection/
├── data/
│ ├── isic/ # Скачанные изображения
│ └── metadata.csv # Метаданные (labels)
├── src/
│ ├── data/
│ │ ├── loader.py # DataLoader для ISIC
│ │ └── init.py
│ └── training/
│ ├── train.py # Training loop
│ └── init.py
├── checkpoints/ # Сохраненные модели
├── requirements.txt # Зависимости
└── README.md

## 🔧 Установка

### 1. Clone репозитория

git clone <https://github.com/m0r0zzk0/medical-melanoma-detection>
cd medical-melanoma-detection

### 2. Создай virtual environment

python -m venv venv
source venv/bin/activate # Linux/Mac

или
.\venv\Scripts\activate # Windows

text

### 3. Установи зависимости

pip install -r requirements.txt

text

### 4. Скачай ISIC dataset

Перейди на <https://www.isic-archive.com/>

Скачай "Challenge 2020: Training" (~24GB)

Распакуй в data/isic/

Скачай metadata.csv в data/

text

## 🚀 Использование

### Обучение модели

python src/training/train.py

text

**Параметры (можешь менять в коде):**

- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001
- `NUM_EPOCHS`: 3 (рекомендуется 10-20 для production)
- `IMG_SIZE`: 224x224

**Output:**

- Обученная модель сохраняется в `checkpoints/best_model_epoch*.pth`
- История обучения в `checkpoints/training_history.json`

### Результаты

После обучения увидишь:
Train Loss: 0.25 | Train Acc: 0.99
Val Loss: 0.18 | Val Acc: 0.98

## 📈 Планы развития (Week 2-3)

- [ ] Data augmentation (medical-safe)
- [ ] Class imbalance handling (weighted loss)
- [ ] Model optimization (quantization, TensorRT)
- [ ] FastAPI endpoint для inference
- [ ] Docker контейнеризация
- [ ] Unit & integration tests

## 🛠️ Технологии

- **PyTorch** - глубокое обучение
- **torchvision** - pre-trained модели
- **OpenCV** - обработка изображений
- **Pandas** - работа с метаданными
- **NumPy** - операции с массивами

## ⚙️ Требования

- Python 3.10+
- NVIDIA GPU (рекомендуется, но не обязательно)
- 25GB+ дискового пространства (для датасета)

## 📝 Лицензия

Этот проект создан в образовательных целях.

## 👤 Автор

Разработано как portfolio project для CV/ML позиции.

---

**Обновлено**: Декабрь 2025
