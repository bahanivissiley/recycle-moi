# 🗑️ Recycle-moi

Application de classification de déchets par deep learning. 
Du training au déploiement sur Play Store.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Objectif

Projet fil rouge pour maîtriser le cycle complet d'un projet IA :
- ✅ Machine Learning (Deep Learning avec PyTorch)
- ✅ MLOps (Versioning, Pipeline, Reproductibilité)
- ✅ Backend API (FastAPI)
- ✅ DevOps (Docker, CI/CD)
- 🔄 Mobile App (Flutter)
- 🔄 Déploiement (Render + Play Store)

## 📊 Résultats

- **Modèle** : ResNet18 fine-tuné
- **Test Accuracy** : **83.55%**
- **Validation Accuracy** : 84.95%
- **Classes** : 7 catégories (cardboard, e-waste, glass, medical, metal, paper, plastic)
- **Dataset** : 17,853 images

### Métriques par Classe

| Classe | Accuracy |
|--------|----------|
| e-waste | 92.05% |
| medical | 85.28% |
| metal | 84.62% |
| glass | 82.28% |
| paper | 81.85% |
| cardboard | 79.34% |
| plastic | 78.36% |

## 🏗️ Architecture
```
recycle-moi/
├── backend-api/          # Backend ML + API
│   ├── src/              # Code source modulaire
│   ├── scripts/          # Scripts CLI
│   └── tests/            # Tests unitaires
├── mobile-app/           # Application Flutter
├── notebooks/            # Notebooks d'expérimentation
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Backend (ML + API)
```bash
# Clone le repo
git clone https://github.com/ton-username/recycle-moi.git
cd recycle-moi/backend-api

# Setup environnement
conda create -n recyclemoi python=3.11 -y
conda activate recyclemoi
pip install -r requirements.txt

# Évaluer le modèle
python scripts/evaluate.py --checkpoint checkpoints/v1.0/best_model.pth

# (API - À venir)
# python api/main.py
```

### Mobile App (À venir)
```bash
cd mobile-app
flutter pub get
flutter run
```

## 📝 Documentation

- [Backend README](backend-api/README.md) - Setup et entraînement
- [API Documentation](docs/api.md) - Endpoints et usage (à venir)
- [Architecture](docs/architecture.md) - Détails techniques (à venir)
- [Déploiement](docs/deployment.md) - Guide de déploiement (à venir)

## 🛠️ Technologies

**Machine Learning**
- PyTorch 2.5.1
- TorchVision
- Transfer Learning (ResNet18)

**Backend** (à venir)
- FastAPI
- Uvicorn
- Pydantic

**DevOps** (à venir)
- Docker
- GitHub Actions
- Render

**Mobile**
- Flutter
- Dart

## 📈 Roadmap

- [x] Phase 1 : Machine Learning (Semaine 1) ✅
  - [x] Setup GPU + Dataset
  - [x] Baseline CNN
  - [x] Transfer Learning
  - [x] Fine-tuning (83.55% accuracy)
  
- [x] Phase 2 : MLOps (Semaine 2) ✅
  - [x] Structuration code
  - [x] Model versioning
  
- [x] Phase 3 : API Backend (Semaine 2) 🔄
  - [x] FastAPI setup
  - [x] Endpoints prédiction
  - [x] Tests
  
- [x] Phase 4 : DevOps (Semaine 3)
  - [x] Dockerisation
  - [x] CI/CD
  - [x] Déploiement Render
  
- [ ] Phase 5 : Mobile App (Semaine 4)
  - [ ] UI Flutter
  - [ ] Intégration API
  - [ ] Tests
  - [ ] Publication Play Store

## 📸 Screenshots

_À venir : captures d'écran de l'app mobile et de l'API_

## 🤝 Contribution

Ce projet est à but éducatif et personnel. 
Suggestions et feedback sont les bienvenus !

## 👨‍💻 Auteur

**Bahani**
- LinkedIn : [Ton profil](https://linkedin.com/in/bahanivissiley)
- GitHub : [@bahanivissiley](https://github.com/bahanivissiley)

## 📄 License

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Dataset de classification de déchets
- PyTorch & TorchVision
- Communauté ML/DL

---

⭐ Si ce projet t'aide dans ton apprentissage, n'hésite pas à mettre une étoile !
