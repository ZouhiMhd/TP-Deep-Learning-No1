#Utiliser une image de base Python
FROM python:3.10-slim

#D finir le r pertoire de travail
WORKDIR /app

#Copier le fichier des d pendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copier le reste de l’application
COPY . .

#Exposer le port de l’application Flask
EXPOSE 5000

#Commande pour d marrer l’’application
CMD ["python", "app.py"]