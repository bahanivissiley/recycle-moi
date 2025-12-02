"""
Script de test complet pour l'API Recycle-moi dans Docker
"""

import requests
import time
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def wait_for_api(max_retries=30, delay=2):
    """Attendre que l'API soit prête"""
    print("⏳ Attente du démarrage de l'API...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API prête!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"   Tentative {i+1}/{max_retries}...")
            time.sleep(delay)
    
    print("❌ L'API n'a pas démarré à temps")
    return False

def test_root():
    """Test endpoint racine"""
    print("\n🧪 Test GET /")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ PASS")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_health():
    """Test health check"""
    print("\n🧪 Test GET /health")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Service: {data.get('service')}")
        assert response.status_code == 200
        assert data.get('status') == 'healthy'
        print("   ✅ PASS")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_model_info():
    """Test model info"""
    print("\n🧪 Test GET /model/info")
    try:
        response = requests.get(f"{API_URL}/model/info")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Version: {data.get('model_version')}")
        print(f"   Architecture: {data.get('architecture')}")
        print(f"   Accuracy: {data.get('test_accuracy')}%")
        print(f"   Classes: {len(data.get('classes', []))} classes")
        assert response.status_code == 200
        assert data.get('test_accuracy') > 0
        print("   ✅ PASS")
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False

def test_predict():
    """Test prédiction"""
    print("\n🧪 Test POST /predict")
    
    # Chercher une image de test
    test_image = None
    possible_paths = [
        "test_image.jpg",
        "../dataset/test/plastic/plastic001.jpg",
        "../dataset/test/glass/glass001.jpg"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if not test_image:
        print("   ⚠️  SKIP: Aucune image de test trouvée")
        return None
    
    try:
        print(f"   Image: {test_image}")
        
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prédiction: {data.get('predicted_class')}")
            print(f"   ✅ Confiance: {data.get('confidence'):.2%}")
            print(f"   ✅ Top 3 classes:")
            
            # Trier par probabilité
            probs = data.get('all_probabilities', {})
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for cls, prob in sorted_probs:
                print(f"      {cls:12s}: {prob:.2%}")
            
            assert data.get('success') == True
            assert 0 <= data.get('confidence') <= 1
            print("   PASS")
            return True
        else:
            print(f"   FAIL: {response.json()}")
            return False
            
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

def test_invalid_file():
    """Test avec un fichier invalide"""
    print("\n🧪 Test POST /predict (fichier invalide)")
    try:
        # Créer un fichier texte temporaire
        files = {'file': ('test.txt', b'Not an image', 'text/plain')}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        print(f"   Status: {response.status_code}")
        assert response.status_code == 400  # Bad Request attendu
        print("   PASS (erreur attendue)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

def main():
    """Lancer tous les tests"""
    print("=" * 60)
    print("🧪 TESTS API RECYCLE-MOI (DOCKER)")
    print("=" * 60)
    
    # Attendre que l'API soit prête
    if not wait_for_api():
        sys.exit(1)
    
    # Lancer les tests
    results = []
    results.append(("Root", test_root()))
    results.append(("Health", test_health()))
    results.append(("Model Info", test_model_info()))
    results.append(("Predict", test_predict()))
    results.append(("Invalid File", test_invalid_file()))
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result == True)
    failed = sum(1 for _, result in results if result == False)
    skipped = sum(1 for _, result in results if result is None)
    
    for name, result in results:
        if result == True:
            status = "PASS"
        elif result == False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"{name:20s}: {status}")
    
    print("-" * 60)
    print(f"Total: {len(results)} tests")
    print(f"Réussis: {passed}")
    print(f"Échoués: {failed}")
    print(f"Ignorés: {skipped}")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 Tous les tests sont passés!")
        sys.exit(0)

if __name__ == '__main__':
    main()