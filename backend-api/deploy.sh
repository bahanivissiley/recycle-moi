#!/bin/bash

# Script de déploiement rapide pour Recycle-moi API

echo "🚀 Déploiement de Recycle-moi API"
echo "=================================="

# Arrêter les conteneurs existants
echo "🛑 Arrêt des conteneurs existants..."
docker-compose down

# Builder l'image
echo "🏗️  Build de l'image Docker..."
docker build -t recyclemoi-api:latest .

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du build"
    exit 1
fi

# Lancer les conteneurs
echo "🚀 Lancement des conteneurs..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du lancement"
    exit 1
fi

# Attendre que l'API démarre
echo "⏳ Attente du démarrage de l'API..."
sleep 10

# Test health check
echo "🧪 Test du health check..."
curl -f http://localhost:8000/health

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ API déployée avec succès!"
    echo "📡 Swagger UI: http://localhost:8000/docs"
    echo "📊 Health: http://localhost:8000/health"
    echo "🔍 Logs: docker-compose logs -f api"
else
    echo ""
    echo "⚠️  API démarrée mais health check échoué"
    echo "🔍 Vérifier les logs: docker-compose logs api"
fi