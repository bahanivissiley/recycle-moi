# 🐳 Docker Guide - Recycle-moi API

Guide pour containeriser et déployer l'API avec Docker.

## 🚀 Quick Start

### Prérequis

- Docker 20.10+
- Docker Compose 2.0+

### Lancer l'API
```bash
# Avec docker-compose (recommandé)
docker-compose up -d

# Voir les logs
docker-compose logs -f api

# Arrêter
docker-compose down
```

L'API sera disponible sur **http://localhost:8000**

## 🏗️ Build

### Build de l'image
```bash
# Build standard
docker build -t recyclemoi-api:latest .

# Build optimisé (CPU-only, plus léger)
docker build -f Dockerfile.optimized -t recyclemoi-api:optimized .
```

### Taille des images

- **Standard (GPU)** : ~2.5GB
- **Optimisée (CPU)** : ~1.5GB

## 📦 Structure Docker
```
Dockerfile              # Image principale
Dockerfile.optimized    # Image optimisée CPU-only
docker-compose.yml      # Orchestration
.dockerignore          # Fichiers exclus
```

## 🔧 Configuration

### Variables d'environnement
```yaml
environment:
  - MODEL_PATH=checkpoints/v1.0/best_model.pth
  - API_PORT=8000
  - LOG_LEVEL=INFO
```

### Volumes
```yaml
volumes:
  # Code (développement avec hot-reload)
  - ./api:/app/api
  - ./src:/app/src
  
  # Modèle (évite de le copier dans l'image)
  - ./checkpoints:/app/checkpoints
```

### Ports

- `8000` : API REST

## 🧪 Tests
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Prédiction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

## 🐛 Debug

### Logs
```bash
# Voir les logs
docker logs recyclemoi-api

# Suivre en temps réel
docker logs -f recyclemoi-api
```

### Entrer dans le conteneur
```bash
docker exec -it recyclemoi-api bash

# Vérifier Python
python --version

# Vérifier PyTorch
python -c "import torch; print(torch.__version__)"
```

## 🚀 Déploiement

### Docker Hub
```bash
# Tag
docker tag recyclemoi-api:latest username/recyclemoi-api:1.0

# Push
docker push username/recyclemoi-api:1.0
```

### Production
```bash
# Lancer en production
docker run -d \
  --name recyclemoi-api \
  -p 8000:8000 \
  --restart unless-stopped \
  -v /path/to/checkpoints:/app/checkpoints \
  recyclemoi-api:latest
```

## 📊 Monitoring

### Health check

Le conteneur inclut un health check automatique :
- Intervalle : 30s
- Timeout : 10s
- Retries : 3
```bash
# Voir le status
docker inspect --format='{{.State.Health.Status}}' recyclemoi-api
```

## 🔒 Sécurité

- ✅ Utilisateur non-root (appuser)
- ✅ Multi-stage build (image minimale)
- ✅ Health checks
- ✅ Restart policy

## 📝 Notes

### GPU Support

Pour utiliser le GPU dans Docker :
```yaml
services:
  api:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Nécessite **nvidia-docker2** installé.

### Performance

- **Temps de build** : 5-10 min (première fois)
- **Taille image** : 1.5-2.5GB
- **Temps de démarrage** : ~10-15s
- **Mémoire** : ~1GB

## 🐛 Troubleshooting

### L'image est trop grosse

→ Utilisez `Dockerfile.optimized` (PyTorch CPU-only)

### Le modèle ne charge pas

→ Vérifiez que le volume `./checkpoints` est bien monté

### Permission denied

→ Le conteneur utilise l'user `appuser` (uid 1000)

### Port déjà utilisé

→ Changez le port dans docker-compose.yml : `"8001:8000"`