import os
import requests
from huggingface_hub import WebhooksServer, WebhookPayload

# Récupération de votre jeton depuis les secrets du Space
HF_TOKEN = os.environ.get("HF_TOKEN")

# Création du serveur d'écoute
app = WebhooksServer()

# Ce décorateur crée une URL se terminant par /webhooks/indexation
@app.add_webhook("/indexation")
async def trigger_indexing(payload: WebhookPayload):
    # On vérifie que c'est bien un dataset qui a été mis à jour
    if payload.repo.type == "dataset" and payload.event.action == "update":
        repo_id = payload.repo.name
        print(f"Nouvelle prise détectée sur {repo_id}. Lancement de l'indexation...")
        
        # Appel à l'API Hugging Face pour forcer le rafraîchissement
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        api_url = f"https://huggingface.co/api/datasets/{repo_id}/viewer"
        
        response = requests.post(api_url, headers=headers)
        
        if response.status_code == 200:
            print(f"Succès : Indexation de {repo_id} en cours.")
        else:
            print(f"Erreur d'indexation : {response.text}")
            
        return {"status": "success", "repo": repo_id}
    
    return {"status": "ignored"}

# Lancement du serveur
if __name__ == "__main__":
    app.launch()